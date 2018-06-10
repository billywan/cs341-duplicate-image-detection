# cs341-duplicate-image-detection
Scalable Duplicate Image Detection project for CS341 at Stanford University in Spring Quarter, 2018.

We designed a 2-stage pipeline consisting of Kernelized Locality Sensitive Hashing (KLSH) and Siamese Network.

We will use the experiment on the Reddit Photoshop Battle dataset of parent as candidate and children as query (original image retrieval) as the running example here.

1. KLSH
KLSH works on Gist descriptor representations of images. To compute gist vectors for candidate and query images, modify paths in the script and run
```
python src/gist-pc.py
```
With the Gist descriptors, we can now run KLSH. KLSH hashes each vector into *b*-bit bit-vectors and divides it into bands of *r*-bits per band. A candidate passes as long as one band hashes into the same bucket as the query. *b* and *r* can be tuned by specifying `-b` and `-r` options of
```
python src/klsh-pc.py
```
which runs KLSH and stores candidate-query pairs in batches ready for the Siamese Network.

2. Siamese Network
We trained 2 Siamese Networks, one based on VGG16 and another based on ResNet50.
To compute predictions on candidate-query batches, run
```
python src/main.py --mode=predict --eval_data_path= --base_model=<vgg16/resnet50> --experiment_name=<path to model parameters folder>
```

Our KLSH+ResNet50 model achieved a >0.5 mean reciprocal rank (MRR) on both original (parent) image retrieval and tampered (child) image retrieval.
