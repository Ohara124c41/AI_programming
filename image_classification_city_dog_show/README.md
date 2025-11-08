# City Dog Show Image Classification

Helper program `data/check_images.py` orchestrates the Udacity "Image Classification for a City Dog Show" workflow:

1. Parse CLI args (`--dir`, `--arch`, `--dogfile`) and time the full run.
2. Build pet labels from filenames, classify each image with the requested CNN (ResNet/AlexNet/VGG), and compare results.
3. Flag whether each label corresponds to a dog using `dognames.txt`, compute accuracy/precision metrics, and print a concise report (with optional misclassification listings).
4. Batch scripts (`run_models_batch.sh`, `run_models_batch_uploaded.sh`) automate running all three models on the provided datasets or on uploaded photos.

Use the hint files inside `data/` for guidance, and install Pillow + PyTorch (CPU wheels are fine) before running `python data/check_images.py --arch vgg`. Runtime results are stored in the `data/` directory or redirected via the batch scripts.
