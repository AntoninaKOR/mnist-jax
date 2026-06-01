import argparse
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random, grad, jit
import flax.linen as nn
import optax
from optax import microbatching
import tensorflow_datasets as tfds

BATCH_SIZE = 32

@dataclass(frozen=True)
class BatchConfig:
    effective_batch_size: int
    microbatch_size: int
    step_size: int

@dataclass(frozen=True)
class TrainMetrics:
    test_accuracy: float
    final_loss: float
    examples_per_sec: float
    train_time_sec: float
    timed_examples: int
    num_epochs: int

class Net(nn.Module):
    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv(features=64, kernel_size=(3, 3))
        self.fc1 = nn.Dense(features=128)
        self.fc2 = nn.Dense(features=10)
        self.dropout = nn.Dropout(rate=0.5)

    def __call__(self, x, training=True, dropout_key=None):
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.dropout(x, deterministic=not training, rng=dropout_key)
        x = self.fc2(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description='Train MNIST classifier with JAX')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help='Effective batch size without accumulation (default: 32)',
    )
    parser.add_argument(
        '--accum-method',
        choices=['none', 'multisteps', 'microbatch'],
        default='none',
        help=(
            'Gradient accumulation strategy: none, optax.MultiSteps, '
            'or microbatching.microbatch (default: none)'
        ),
    )
    parser.add_argument(
        '--gradient-accumulation',
        action='store_true',
        help='Alias for --accum-method multisteps',
    )
    parser.add_argument(
        '--accum-steps',
        type=int,
        default=4,
        help='Micro-batches to accumulate before each optimizer update (default: 4)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs (default: 5)',
    )
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=1,
        help='Epochs excluded from FPS timing while JIT compiles (default: 1)',
    )
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='Train all accumulation methods and compare accuracy with FPS',
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark all accumulation methods and print examples/s',
    )
    parser.add_argument(
        '--benchmark-steps',
        type=int,
        default=100,
        help='Optimizer updates per method in benchmark mode (default: 100)',
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=10,
        help='Warmup steps before timing in benchmark mode (default: 10)',
    )
    return parser.parse_args()

def resolve_accum_method(args):
    if args.gradient_accumulation:
        if args.accum_method != 'none':
            raise ValueError('Use either --gradient-accumulation or --accum-method, not both')
        return 'multisteps'
    return args.accum_method

def resolve_batch_config(
    batch_size: int,
    accum_method: str,
    accum_steps: int,
) -> BatchConfig:
    if batch_size < 1:
        raise ValueError('--batch-size must be >= 1')
    if accum_method == 'none':
        return BatchConfig(
            effective_batch_size=batch_size,
            microbatch_size=batch_size,
            step_size=batch_size,
        )
    if batch_size % accum_steps != 0:
        raise ValueError(
            f'--batch-size ({batch_size}) must be divisible by '
            f'--accum-steps ({accum_steps})'
        )
    microbatch_size = batch_size // accum_steps
    if accum_method == 'microbatch':
        step_size = batch_size
    else:
        step_size = microbatch_size
    return BatchConfig(
        effective_batch_size=batch_size,
        microbatch_size=microbatch_size,
        step_size=step_size,
    )

def make_loss_fn(model):
    def loss_fn(params, images, labels, dropout_key):
        logits = model.apply(
            params, images, training=True, dropout_key=dropout_key)
        one_hot = jax.nn.one_hot(labels, 10)
        return jnp.mean(optax.softmax_cross_entropy(
            logits=logits, labels=one_hot))
    return loss_fn

def make_grad_fn(loss_fn, accum_method, microbatch_size):
    if accum_method == 'microbatch':
        return microbatching.microbatch(
            grad(loss_fn),
            argnums=(1, 2),
            microbatch_size=microbatch_size,
            accumulator=microbatching.AccumulationType.MEAN,
        )
    return grad(loss_fn)

def make_optimizer(accum_method, accum_steps):
    base_tx = optax.adam(0.001)
    if accum_method == 'multisteps':
        return optax.MultiSteps(base_tx, every_k_schedule=accum_steps)
    return base_tx

def make_train_step(grad_fn, loss_fn, tx):
    @jit
    def train_step(params, opt_state, images, labels, dropout_key):
        loss = loss_fn(params, images, labels, dropout_key)
        grads = grad_fn(params, images, labels, dropout_key)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step

def init_training_state(
    model,
    accum_method,
    accum_steps,
    batch_config,
    key,
    sample_images,
):
    rngs = {'params': key, 'dropout': key}
    params = model.init(rngs, sample_images)
    loss_fn = make_loss_fn(model)
    grad_fn = make_grad_fn(
        loss_fn, accum_method, microbatch_size=batch_config.microbatch_size)
    tx = make_optimizer(accum_method, accum_steps)
    opt_state = tx.init(params)
    return params, opt_state, grad_fn, tx, loss_fn

def make_fixed_batches(train_images, train_labels, batch_size, num_batches):
    batches = []
    for batch_idx in range(num_batches):
        start = (batch_idx * batch_size) % (len(train_images) - batch_size)
        batches.append((
            train_images[start:start + batch_size],
            train_labels[start:start + batch_size],
        ))
    return batches

def run_train_step(train_step, params, opt_state, images, labels, key):
    key, dropout_key = random.split(key)
    params, opt_state, loss = train_step(
        params, opt_state, images, labels, dropout_key)
    return params, opt_state, key, loss

def evaluate(model, params, test_images, test_labels):
    test_logits = model.apply(params, test_images, training=False)
    test_predictions = jnp.argmax(test_logits, axis=-1)
    return float(jnp.mean(test_predictions == test_labels))

def benchmark_method(
    model,
    accum_method,
    accum_steps,
    batch_config,
    train_images,
    train_labels,
    key,
    num_steps,
    warmup_steps,
):
    params, opt_state, grad_fn, tx, loss_fn = init_training_state(
        model,
        accum_method,
        accum_steps,
        batch_config,
        key,
        train_images[0:1],
    )
    train_step = make_train_step(grad_fn, loss_fn, tx)

    if accum_method == 'none':
        single_batches = make_fixed_batches(
            train_images,
            train_labels,
            batch_config.effective_batch_size,
            warmup_steps + num_steps,
        )

        def run_single_steps(start_idx, count):
            nonlocal params, opt_state, key
            for step in range(count):
                images, labels = single_batches[start_idx + step]
                params, opt_state, key, _ = run_train_step(
                    train_step, params, opt_state, images, labels, key)
            params, opt_state = jax.block_until_ready((params, opt_state))

        run_single_steps(0, warmup_steps + num_steps)
        start = time.perf_counter()
        run_single_steps(warmup_steps, num_steps)
        elapsed = time.perf_counter() - start
    elif accum_method == 'multisteps':
        num_micro_steps = (warmup_steps + num_steps) * accum_steps
        micro_batches = make_fixed_batches(
            train_images,
            train_labels,
            batch_config.microbatch_size,
            num_micro_steps,
        )

        def run_micro_steps(start_idx, count):
            nonlocal params, opt_state, key
            for step in range(count):
                images, labels = micro_batches[start_idx + step]
                params, opt_state, key, _ = run_train_step(
                    train_step, params, opt_state, images, labels, key)
            params, opt_state = jax.block_until_ready((params, opt_state))

        run_micro_steps(0, num_micro_steps)
        start = time.perf_counter()
        run_micro_steps(warmup_steps * accum_steps, num_steps * accum_steps)
        elapsed = time.perf_counter() - start
    else:
        full_batches = make_fixed_batches(
            train_images,
            train_labels,
            batch_config.effective_batch_size,
            warmup_steps + num_steps,
        )

        def run_full_steps(start_idx, count):
            nonlocal params, opt_state, key
            for step in range(count):
                images, labels = full_batches[start_idx + step]
                params, opt_state, key, _ = run_train_step(
                    train_step, params, opt_state, images, labels, key)
            params, opt_state = jax.block_until_ready((params, opt_state))

        run_full_steps(0, warmup_steps + num_steps)
        start = time.perf_counter()
        run_full_steps(warmup_steps, num_steps)
        elapsed = time.perf_counter() - start

    total_examples = num_steps * batch_config.effective_batch_size
    examples_per_sec = total_examples / elapsed
    ms_per_effective_step = (elapsed / num_steps) * 1000
    return examples_per_sec, ms_per_effective_step

def run_benchmark(args, train_images, train_labels, key):
    accum_steps = args.accum_steps
    batch_config = resolve_batch_config(args.batch_size, 'multisteps', accum_steps)
    methods = ['none', 'multisteps', 'microbatch']
    results = []

    print(
        f'Benchmark: {args.benchmark_steps} steps, '
        f'effective_batch={batch_config.effective_batch_size}, '
        f'microbatch={batch_config.microbatch_size}, '
        f'accum_steps={accum_steps}'
    )
    print(f'{"Method":<12} {"Examples/s":>12} {"ms/eff-step":>12}')
    print('-' * 38)

    for method in methods:
        key, method_key = random.split(key)
        method_batch_config = resolve_batch_config(
            args.batch_size, method, accum_steps)
        examples_per_sec, ms_per_effective_step = benchmark_method(
            Net(),
            method,
            accum_steps,
            method_batch_config,
            train_images,
            train_labels,
            method_key,
            args.benchmark_steps,
            args.warmup_steps,
        )
        results.append((method, examples_per_sec, ms_per_effective_step))
        print(f'{method:<12} {examples_per_sec:>12.1f} {ms_per_effective_step:>12.2f}')

    print()
    print('Notes:')
    print('- All methods use the same effective batch size per optimizer update')
    print(
        f'- Effective batch: {batch_config.effective_batch_size}; '
        f'microbatch with accumulation: {batch_config.microbatch_size}'
    )
    print('- none: one forward/backward on the full effective batch')
    print('- multisteps/microbatch: smaller micro-batches accumulated to the same effective batch')
    print('- ms/eff-step: time per effective-batch optimizer update')

    return results

def train(
    model,
    params,
    opt_state,
    grad_fn,
    tx,
    loss_fn,
    train_images,
    train_labels,
    test_images,
    test_labels,
    key,
    accum_method,
    batch_config,
    num_epochs,
    warmup_epochs,
):
    if accum_method != 'none':
        print(
            f'Gradient accumulation ({accum_method}): '
            f'microbatch={batch_config.microbatch_size}, '
            f'effective batch={batch_config.effective_batch_size}'
        )
    else:
        print(f'Batch size: {batch_config.effective_batch_size}')

    step_size = batch_config.step_size
    train_step = make_train_step(grad_fn, loss_fn, tx)
    timed_examples = 0
    final_loss = 0.0
    train_start = None

    for epoch in range(num_epochs):
        epoch_examples = 0
        for i in range(0, len(train_images), step_size):
            batch_images = train_images[i:i + step_size]
            batch_labels = train_labels[i:i + step_size]
            batch_examples = int(batch_images.shape[0])
            if batch_examples == 0:
                continue

            params, opt_state, key, loss = run_train_step(
                train_step, params, opt_state, batch_images, batch_labels, key)
            final_loss = float(loss)
            epoch_examples += batch_examples

            if i % (step_size * 100) == 0:
                print(f'Epoch {epoch + 1}, Loss: {final_loss}')

        params, opt_state = jax.block_until_ready((params, opt_state))

        if epoch >= warmup_epochs:
            if train_start is None:
                train_start = time.perf_counter()
            timed_examples += epoch_examples

        print(f'Epoch {epoch + 1}/{num_epochs} complete')

    train_time_sec = (
        time.perf_counter() - train_start if train_start is not None else 0.0
    )
    examples_per_sec = timed_examples / train_time_sec if train_time_sec > 0 else 0.0

    test_accuracy = evaluate(model, params, test_images, test_labels)
    metrics = TrainMetrics(
        test_accuracy=test_accuracy,
        final_loss=final_loss,
        examples_per_sec=examples_per_sec,
        train_time_sec=train_time_sec,
        timed_examples=timed_examples,
        num_epochs=num_epochs,
    )

    print(
        f'Training FPS: {metrics.examples_per_sec:.1f} examples/s '
        f'({metrics.timed_examples} examples in {metrics.train_time_sec:.2f}s, '
        f'after {warmup_epochs} warmup epoch(s))'
    )
    print(f'Test accuracy: {metrics.test_accuracy * 100:.2f}%')

    return params, key, metrics

def run_training_run(
    accum_method,
    args,
    train_images,
    train_labels,
    test_images,
    test_labels,
    key,
):
    batch_config = resolve_batch_config(
        args.batch_size, accum_method, args.accum_steps)
    model = Net()
    params, opt_state, grad_fn, tx, loss_fn = init_training_state(
        model,
        accum_method,
        args.accum_steps,
        batch_config,
        key,
        train_images[0:1],
    )

    print(f'\n=== {accum_method} ===')
    _, _, metrics = train(
        model,
        params,
        opt_state,
        grad_fn,
        tx,
        loss_fn,
        train_images,
        train_labels,
        test_images,
        test_labels,
        key,
        accum_method,
        batch_config,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )
    return metrics

def run_compare_all(args, train_images, train_labels, test_images, test_labels, key):
    accum_steps = args.accum_steps
    batch_config = resolve_batch_config(args.batch_size, 'multisteps', accum_steps)
    methods = ['none', 'multisteps', 'microbatch']
    results = []

    print(
        f'Full training compare: epochs={args.epochs}, '
        f'effective_batch={batch_config.effective_batch_size}, '
        f'microbatch={batch_config.microbatch_size}, '
        f'accum_steps={accum_steps}, '
        f'warmup_epochs={args.warmup_epochs}'
    )

    for method in methods:
        key, method_key = random.split(key)
        metrics = run_training_run(
            method,
            args,
            train_images,
            train_labels,
            test_images,
            test_labels,
            method_key,
        )
        results.append((method, metrics))

    print(f'\n{"Method":<12} {"Examples/s":>12} {"Accuracy":>10} {"Time (s)":>10}')
    print('-' * 48)
    for method, metrics in results:
        print(
            f'{method:<12} '
            f'{metrics.examples_per_sec:>12.1f} '
            f'{metrics.test_accuracy * 100:>9.2f}% '
            f'{metrics.train_time_sec:>10.2f}'
        )

    return results

def load_data():
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(
        ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))

    train_images = jnp.float32(train_ds['image']) / 255.0
    train_labels = train_ds['label']
    test_images = jnp.float32(test_ds['image']) / 255.0
    test_labels = test_ds['label']
    return train_images, train_labels, test_images, test_labels

def main():
    args = parse_args()
    accum_method = resolve_accum_method(args)
    if args.accum_steps < 1:
        raise ValueError('--accum-steps must be >= 1')
    if args.batch_size < 1:
        raise ValueError('--batch-size must be >= 1')
    if args.epochs < 1:
        raise ValueError('--epochs must be >= 1')
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError('--warmup-epochs must be >= 0 and < --epochs')
    if args.benchmark_steps < 1:
        raise ValueError('--benchmark-steps must be >= 1')
    if args.warmup_steps < 0:
        raise ValueError('--warmup-steps must be >= 0')

    key = random.PRNGKey(0)
    train_images, train_labels, test_images, test_labels = load_data()

    if args.benchmark:
        run_benchmark(args, train_images, train_labels, key)
        return

    if args.compare_all:
        run_compare_all(
            args, train_images, train_labels, test_images, test_labels, key)
        return

    key, train_key = random.split(key)
    run_training_run(
        accum_method,
        args,
        train_images,
        train_labels,
        test_images,
        test_labels,
        train_key,
    )

if __name__ == '__main__':
    main()
