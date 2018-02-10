FROM maxim_pythia_mill_container
# Add Tini
ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

RUN git clone https://github.com/scikit-optimize/scikit-optimize.git
WORKDIR /usr/app/scikit-optimize
RUN pip install -e.

WORKDIR /usr/app
COPY . /usr/app/

RUN pip install -r env_requirements.txt

ENTRYPOINT ["/tini", "--"]
# Run your program under Tini
CMD ["jupyter", "notebook", "--no-browser", "--port=8888", "--ip=0.0.0.0", "--allow-root"]



