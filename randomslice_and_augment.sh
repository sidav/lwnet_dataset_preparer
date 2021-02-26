echo "Making random slices..."
python3 slice.py
echo "Copying backup files"
mkdir previous
cp -r marked previous/
cp -r raw previous/
echo "Copying sliced files to dataset folder..."
rm -r marked/*
rm -r raw/*
cp -r sliced/marked/* marked/
cp sliced/raw/* raw/
echo "Creating the dataset..."
python3 slice_and_augment.py
echo "Done."
