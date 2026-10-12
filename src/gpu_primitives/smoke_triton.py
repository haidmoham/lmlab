import torch
import triton
import triton.language as tl


@triton.jit
def add_one_kernel(values, output, value_count: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.arange(0, block_size)
    mask = offsets < value_count
    input_values = tl.load(values + offsets, mask=mask)
    tl.store(output + offsets, input_values + 1.0, mask=mask)


def main() -> None:
    value_count = 4096
    values = torch.arange(value_count, device="cuda", dtype=torch.float32)
    output = torch.empty_like(values)

    add_one_kernel[(1,)](
        values,
        output,
        value_count=value_count,
        block_size=triton.next_power_of_2(value_count),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, values + 1.0)
    print(f"triton smoke passed on {value_count} values")


if __name__ == "__main__":
    main()
