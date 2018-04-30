FROM registry.ng.bluemix.net/ciml/ciml-base:1

RUN apt-get update && apt-get install -y nginx python3-dev supervisor && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install uwsgi

# forward request and error logs to docker log collector
RUN ln -sf /dev/stdout /var/log/nginx/access.log \
&& ln -sf /dev/stderr /var/log/nginx/error.log
# Run nginx in the foreground
RUN echo "daemon off;" >> /etc/nginx/nginx.conf
# Remove default configuration from Nginx
RUN rm -f /etc/nginx/conf.d/default.conf

EXPOSE 80

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
#COPY ciml-api.conf /etc/nginx/sites-available/ciml-api.conf
COPY ciml-api.conf /etc/nginx/conf.d/ngnix.conf
COPY ciml-uwsgi-api.ini /etc/ciml-uwsgi-api.ini
COPY ciml-train.conf /etc/ciml-train.conf

#RUN ln -s /etc/nginx/sites-available/ciml-api.conf /etc/nginx/sites-enabled/ciml-api.conf
RUN mkdir -p /var/run/uwsgi/

ENV MPLBACKEND=agg
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY run.sh /root/run.sh

CMD /root/run.sh
