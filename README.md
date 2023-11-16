# Different types about fusion

- ### align & align_shuffle
    **e.g.** image_feature = [1, 2, 3], text_feature=[2, 3, 4]
    
    Then the $feature_{fusion}=[1*2, 2*3, 3*4]$.
    
    **shape = (batch_size, d)**
- ### concat
    Simply concat image_feature and text_feature.

    **shape = (batch_size, 2*d)**
- ### cross
    FIM(Feature Interaction Matrix)

    FIM = image_feature $\bigotimes$ text_feature.
    
    Then flatten FIM to the vector which shape is $d^2$.
    
    Use torch.bmm(), which is a batch multiple function of tensor, that means batch-wise outer product of last two dimension.
    
    Because of $Dimension_{min}=3$, the features of image or text must be extended to 3d.
    
    The shape of extended features of image or text is (batch_size, 1, d) or (batch_size, d, 1) which is facilitating for doing outer product. 

    **shape = (batch_size, $d^2$)**
    
- ### cross_nd
    After doing cross, pick diagonal element of FIM.

    **shape = (batch_size, d)**
- ### align_concat
    Concat three parts, include flatten FIM, image_feature and text_feature
    
    shape = (batch_size, 3*d)
- ### attention_m
    Omitted.