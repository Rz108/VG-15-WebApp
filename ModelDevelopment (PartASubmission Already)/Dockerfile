FROM tensorflow/serving

# Copy the model development directory to the container
COPY /ModelDevelopment /

# Copy the models.config file to the container from the ModelDevelopment directory
COPY ModelDevelopment/models/img_classifier/models.config /models.config

# Expose ports
EXPOSE 8500
EXPOSE 8501

# Set up the entrypoint script
RUN echo '#!/bin/bash \n\n\
tensorflow_model_server \
--rest_api_port=$PORT \
--model_config_file=/models.config \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh
