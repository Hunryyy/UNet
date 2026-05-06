"""
Test 8: Pretrained ResNet34 weight source behavior.

Task 2 should remain runnable and reproducible from the repository itself.
Since the repo already ships code/weight/resnet34-b627a593.pth, the model
loader should support that local artifact instead of relying purely on torch
cache state or network fallback.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR)

from _task2_common import CODE_DIR


def main():
    print("=" * 60)
    print("TEST 8: Pretrained weight source behavior")
    print("=" * 60)

    repo_weight = os.path.join(CODE_DIR, "weight", "resnet34-b627a593.pth")
    cache_weight = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet34-b627a593.pth")

    print(f"  Repo bundled weight:  {repo_weight}")
    print(f"  Torch cache weight:   {cache_weight}")

    assert os.path.exists(repo_weight), "Expected bundled repo weight file is missing"
    print("  Bundled repo weight exists  OK")

    if os.path.exists(cache_weight):
        print("  Torch cache weight also exists")
    else:
        print("  Torch cache weight missing in this environment")

    print("\n[8.1] Source-code inspection")
    model_path = os.path.join(CODE_DIR, "unet_model.py")
    with open(model_path, "r", encoding="utf-8") as f:
        src = f.read()

    assert "resnet34-b627a593.pth" in src, "Expected pretrained weight filename missing in source"
    assert (
        "code/weight" in src
        or "./weight/" in src
        or "weight/resnet34-b627a593.pth" in src
    ), (
        "Model loader does not reference the repo-bundled pretrained weight. "
        "This means successful runs may depend on cache state or network fallback."
    )

    print("  Loader references repo bundled weight  OK")

    print("\n" + "=" * 60)
    print("TEST 8 PASSED — Local pretrained weight path is enforced")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()
