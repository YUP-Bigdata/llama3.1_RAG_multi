ARG ELK_VERSION

# https://www.docker.elastic.co/
FROM docker.elastic.co/elasticsearch/elasticsearch:${ELK_VERSION}

COPY analysis-nori-8.15.0.zip /tmp/analysis-nori.zip

RUN bin/elasticsearch-plugin install file:///tmp/analysis-nori.zip
# RUN bin/elasticsearch-plugin install analysis-nori
# Add your elasticsearch plugins setup here
# Example: RUN elasticsearch-plugin install analysis-icu
