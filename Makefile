.PHONY: dev dev-subpath clean

dev:
	hugo server --bind 0.0.0.0 --baseURL http://localhost:1313/ --appendPort=false

dev-subpath:
	hugo server --bind 0.0.0.0 --baseURL http://localhost:1313/book/ --appendPort=false

clean:
	rm -rf public resources
