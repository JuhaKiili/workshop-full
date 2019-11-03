docker run --runtime=nvidia -t -i -v "$PWD":/work -p 8888:8888 -p 6006:6006 juhavalohai/workshop:tf-gpu2 sh /work/$1.sh

