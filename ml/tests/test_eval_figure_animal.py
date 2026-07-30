"""The figure/animal boundary measurement (ml.md#the-figureanimal-boundary).

The network-dependent half (shard sampling, gold lookup) is exercised by running
the script; these cover the counting logic the documented numbers rest on, since a
miscounted bucket would silently change a recorded conclusion rather than fail.
"""

from eval_figure_animal import (
    ambiguous_objects,
    keyword_bucket,
    measure_reach,
    measure_validity,
)


def annotation(name: str, *tag_names: str, categories: tuple[str, ...] = ()) -> dict:
    """An objaverse-shaped annotation: title, tags, and Sketchfab categories."""
    return {
        "name": name,
        "tags": [{"name": tag} for tag in tag_names],
        "categories": [{"name": category} for category in categories],
    }


CHARACTERS = ("characters-creatures",)


def test_only_the_contested_category_is_collected() -> None:
    """`characters-creatures` gates to both classes; `animals-pets` gates to one, so
    it is already decided and no figure/animal rule applies to it."""
    contested = annotation("cat person", "character", categories=CHARACTERS)
    decided = annotation("tabby", "cat", categories=("animals-pets",))

    gated = ambiguous_objects({"contested": contested, "decided": decided})

    assert list(gated) == ["contested"]


def test_buckets_split_by_which_lists_fire() -> None:
    assert keyword_bucket(annotation("a", "character", "cat")) == "both"
    assert keyword_bucket(annotation("a", "character")) == "figure only"
    assert keyword_bucket(annotation("a", "cat")) == "animal only"
    assert keyword_bucket(annotation("a", "zbrush")) == "neither"


def test_title_tokens_count_as_hits_like_the_resolver() -> None:
    """The measurement must tokenize titles too, or it describes a different
    function than the resolver it is grading."""
    assert keyword_bucket(annotation("robot dragon")) == "both"


def test_reach_reports_the_share_a_precedence_rule_could_change() -> None:
    """Only "both" is reachable — that share is the ceiling on any stance rule."""
    objects = {
        "both": annotation("cat character", categories=CHARACTERS),
        "figure": annotation("knight", categories=CHARACTERS),
        "animal": annotation("wolf", categories=CHARACTERS),
        "neither": annotation("untitled", "blender", categories=CHARACTERS),
    }

    reach = measure_reach(objects)

    assert reach["ambiguous_count"] == 4
    assert reach["both_lists_fire_share"] == 0.25
    assert reach["neither_fires_share"] == 0.25
    # An object matching neither list is left unlabeled, not guessed.
    assert reach["by_bucket_and_outcome"]["neither -> ambiguous"] == 1


def test_token_skew_ignores_tokens_below_the_support_floor() -> None:
    """One uploader's tagging habit is not a signal, so a token under
    MIN_GOLD_SUPPORT is excluded however lopsided it looks."""
    objects = {
        f"uid{index}": annotation("thing", "rare", categories=CHARACTERS)
        for index in range(4)
    }
    uid_to_gold_class = dict.fromkeys(objects, "figure")

    validity = measure_validity(objects, uid_to_gold_class)

    assert validity["gold_count"] == 4
    assert validity["token_skew"] == []


def test_token_skew_ranks_a_discriminating_token_above_a_neutral_one() -> None:
    objects = {}
    uid_to_gold_class = {}
    # Eight per class, so "beast" clears MIN_GOLD_SUPPORT on the animals alone.
    for index in range(8):
        # "shared" appears on every object; "beast" only on the animals.
        for gold_class in ("figure", "animal"):
            uid = f"{gold_class}{index}"
            extra = ("beast",) if gold_class == "animal" else ()
            objects[uid] = annotation("thing", "shared", *extra, categories=CHARACTERS)
            uid_to_gold_class[uid] = gold_class

    validity = measure_validity(objects, uid_to_gold_class)
    skew_by_token = {row["token"]: row for row in validity["token_skew"]}

    assert skew_by_token["shared"]["figure_share"] == 0.5
    assert skew_by_token["beast"]["figure_share"] == 0.0
    assert validity["token_skew"][0]["token"] == "beast"


def test_abstention_is_reported_separately_from_being_wrong() -> None:
    """The headline finding on this boundary is that the labeler mostly declines to
    answer, which a precision number hides."""
    objects = {
        "silent": annotation("untitled", "blender", categories=CHARACTERS),
        "answered": annotation("knight", categories=CHARACTERS),
    }
    uid_to_gold_class = {"silent": "figure", "answered": "figure"}

    validity = measure_validity(objects, uid_to_gold_class)

    assert validity["abstained"] == 1
    assert validity["abstention_share"] == 0.5
    assert validity["gold_to_weak"]["figure -> ambiguous"] == 1
    assert validity["gold_to_weak"]["figure -> figure"] == 1
