# Hate-UDF: Explainable Hateful Meme Detection with Uncertainty-aware Dynamic Fusion

This project is part of the paper [Hate-UDF: Explainable Hateful Meme Detection with Uncertainty-aware Dynamic Fusion]().

## Install

- In `CUDA v11.3` environment, use `Python v3.8.10` and `PyTorch v1.11.0`
- Most of the required packages and version numbers are already written in `requirements.txt`

## Datasets

- The download address for each benchmark used has been placed in the `download`file (under `/data`) at the appropriate location of the project.

## Run

- Use the following command to get the results of the experiment.

```shell
bash ./run.sh
```

## Uncertainty-aware Dynamic Fusion

- ### `align & align_shuffle`

  **e.g.** image_feature = [1, 2, 3], text_feature=[2, 3, 4]

  Then the $feature_{fusion}=[1\times2,\space2\times3,\space3\times4]$.

  **shape = (batch_size, d)**

- ### `concat`

  Simply `concat` image_feature and text_feature.

  **shape = (batch_size, 2*d)**

- ### `cross`

  **FIM (Feature Interaction Matrix)**

  FIM = image_feature $\bigotimes$ text_feature.

  Then flatten FIM to the vector which shape is $d^2$.

  Use `torch.bmm()`, which is a batch multiple function of tensor, that means batch-wise outer product of last two dimension.

  Because of $Dimension_{min}=3$, the features of image or text must be extended to 3-dim.

  The shape of extended features of image or text is **(batch_size, 1, d)** or **(batch_size, d, 1)** which is facilitating for doing outer product. 

  **shape = (batch_size, $d^2$)**

- ### `cross_nd`

  After doing cross, pick diagonal element of FIM.

  **shape = (batch_size, d)**

- ### `align_concat`

  `concat` three parts, include element-wise product of image_feature and text_feature, image_feature and text_feature

  **shape = (batch_size, 3*d)**

- ### `attention_m`

  Omitted.
