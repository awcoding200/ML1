# ML1
Machine learning 1 homework


## Getting started with the Programming Homeworks
The course assumes that you use a recent linux distribution with Python 3.0 installed, or other operating system supporting Python. (Windows users could install setuptools as an alternative.)

### Installing required Python tools libraries

If not installed, run
```
sudo apt-get install jupyter-notebook
sudo apt-get install python3-numpy
sudo apt-get install python3-scipy
sudo apt-get install python3-matplotlib
sudo apt-get install python3-cvxopt
```
or equivalent command depending on your distribution.

### Installing GPG

As an extra security layer, the distributed code is signed with GPG. If not installed, run
```
sudo apt-get install gnupg
```
or equivalent command depending on your distribution.

### Using our public GPG key

Download our GPG public key from ISIS.
Import it and verify that the fingerprint is:  9AA7 2C0B A06F 34D5 803C 68D1 AA15 341D 98C7 12B6
```
gpg --import public.key
gpg --fingerprint
```
Opening a homework

Download the programming assignment from ISIS
Run the following commands:
```
gpg --output sheet1.zip --decrypt sheet1.zip.gpg
unzip sheet1.zip
Start the notebook
jupyter notebook
```
Zuletzt geändert: Mittwoch, 23. Oktober 2024, 14:13