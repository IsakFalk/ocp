
# Table of Contents

-   [Scripts](#org76f916e)
    -   [`download_pretrained_models.py`](#org096ddb8)
-   [Load and use model](#org04842ef)

This page summarizes the extra changes and additions that we (Isak Falk, Pietro Novelli, Luigi Bonati) has done to the original OCP repository.


<a id="org76f916e"></a>

# Scripts


<a id="org096ddb8"></a>

## `download_pretrained_models.py`

Download the pretrained checkpoint and corresponding config and scale files to the correct directories. Run

    python scripts/download_pretrained_models.py --help

to see how to do this.

The following example downloads the model `CGCNN` trained on the `All` split of the task `S2EF` optimized for the `efwt` metric

    python scripts/download_pretrained_models.py --task s2ef_efwt --split All --model CGCNN


<a id="org04842ef"></a>

# Load and use model

See [playground.py](playground.py) for an example for how to do this.

