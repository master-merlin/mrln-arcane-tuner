"""The held-out split for training experiments: by source image, the same for every arm.

One image path with all its repeats, buckets and captions is one source. Near-identical
images are linked, because a held-out photo whose near-copy trains says little, and a
group is a connected component of those links. The split depends only on the files and
``split_seed``: not on the training seed, the order files arrive in, or other settings.
"""
import hashlib
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image

NEAR_IDENTICAL_BITS = 6
_HASH_WIDTH = 8

Item = dict[str, Any]


@dataclass(frozen=True)
class ValidationSplit:
    training_items: list[Item]
    holdout_keys: list[str]
    representatives: list[Item]
    groups: list[list[str]]


def difference_hash(path: str) -> int:
    """A 64-bit difference hash of the image at ``path``."""
    with Image.open(path) as image:
        small = image.convert("L").resize((_HASH_WIDTH + 1, _HASH_WIDTH), Image.Resampling.LANCZOS)
    pixels = small.tobytes()
    bits = 0
    for row in range(_HASH_WIDTH):
        start = row * (_HASH_WIDTH + 1)
        for col in range(_HASH_WIDTH):
            bits = (bits << 1) | int(pixels[start + col] > pixels[start + col + 1])
    return bits


def _source_key(item: Item) -> str:
    return f"{item['dataset_name']}/{item['id']}".replace("\\", "/")


def _groups(keys_by_path: dict[str, str], image_hash: Callable[[str], int]) -> list[list[str]]:
    """Connected components over near-identical hashes, each sorted, in key order."""
    paths = sorted(keys_by_path, key=keys_by_path.__getitem__)
    hashes = [image_hash(path) for path in paths]
    parent = list(range(len(paths)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if (hashes[i] ^ hashes[j]).bit_count() <= NEAR_IDENTICAL_BITS:
                parent[find(j)] = find(i)

    components: dict[int, list[str]] = {}
    for i, path in enumerate(paths):
        components.setdefault(find(i), []).append(keys_by_path[path])
    return sorted(sorted(group) for group in components.values())


def _representative(items: Iterable[Item]) -> Item:
    """Largest bucket by pixels; then the smallest bucket name; then the first caption."""

    def rank(item: Item) -> tuple:
        width, height = item["target_w"], item["target_h"]
        return (-(width * height), f"{width}x{height}", item.get("caption") or "")

    return min((it for it in items if it.get("variant", "original") == "original"), key=rank)


def _representatives(items: Sequence[Item], keys: Iterable[str]) -> list[Item]:
    by_key: dict[str, list[Item]] = {}
    for item in items:
        by_key.setdefault(_source_key(item), []).append(item)
    return [{**_representative(by_key[key]), "key": key} for key in sorted(keys)]


def split_holdout(
    items: Sequence[Item],
    fraction: float,
    split_seed: int,
    image_hash: Callable[[str], int] = difference_hash,
) -> ValidationSplit:
    """Hold out the first groups, in hash order, reaching ``fraction`` of the source images."""
    keys_by_path = {item["path"]: _source_key(item) for item in items}
    groups = _groups(keys_by_path, image_hash)
    target = math.ceil(fraction * len(keys_by_path))

    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{split_seed}:{group[0]}".encode()).hexdigest(),
    )
    held: list[str] = []
    for group in ordered:
        if len(held) >= target:
            break
        held.extend(group)

    if not held or len(held) == len(keys_by_path):
        names = ", ".join(sorted({str(item["dataset_name"]) for item in items}))
        raise ValueError(
            f"Validation can't split {names}: {len(keys_by_path)} source image(s) in "
            f"{len(groups)} group(s) of near-identical images would leave no images on "
            "one side. Use more images, or a separate validation dataset."
        )

    held_set = set(held)
    return ValidationSplit(
        training_items=[item for item in items if _source_key(item) not in held_set],
        holdout_keys=sorted(held),
        representatives=_representatives(items, held),
        groups=groups,
    )


def split_separate_dataset(
    training_items: Sequence[Item], validation_items: Sequence[Item]
) -> ValidationSplit:
    """Training keeps everything; the named dataset is the validation set."""
    if not validation_items:
        raise ValueError("The validation dataset has no usable images.")
    trained = {item["path"] for item in training_items}
    for item in validation_items:
        if item["path"] in trained:
            raise ValueError(
                f"{item['path']} is in both the training data and the validation dataset."
            )
    keys = {_source_key(item) for item in validation_items}
    return ValidationSplit(
        training_items=list(training_items),
        holdout_keys=sorted(keys),
        representatives=_representatives(validation_items, keys),
        groups=[[key] for key in sorted(keys)],
    )
