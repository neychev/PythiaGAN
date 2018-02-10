# PythiaGAN
Research aimed to simulate experimental data with GANs with distributed parallel generators.

The provided code uses pythia-mill lib: https://github.com/maxim-borisyak/pythia-mill

Everything is avalable within docker. To build docker on your machine:

1. Build Dockerfile from pythia-mill lib mentioned above and tag it with some name (e.g. pythia_mill_image).

2. Check that the name you have chosen is the same as the name of base image in Dockerfile:
```Dockerfile
FROM pythia_mill_image
```
and build the docker image (in this directory):
```bash
docker build .
```
Optionally, tag this image with some name `<IMAGE NAME>`.

3. After build is complete, run docker container from this image connected to some local port:
```bash
docker run -p <YOUR LOCAL PORT>:8888 -ti <IMAGE NAME>
```

Now jupyter should be available on the selected port.
