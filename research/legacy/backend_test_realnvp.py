from realnvp_inference import load_realnvp

model, mean, std = load_realnvp()

print()
print("TEST PASSED")
print("Mean shape:", tuple(mean.shape))
print("Std shape :", tuple(std.shape))
print(
    "Parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    ),
)