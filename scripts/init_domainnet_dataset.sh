TARGET=data/domainnet
mkdir -p $TARGET
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt
wget -P $TARGET -nc http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt
unzip -n -d $TARGET $TARGET/clipart.zip 
unzip -n -d $TARGET $TARGET/infograph.zip
unzip -n -d $TARGET $TARGET/painting.zip
unzip -n -d $TARGET $TARGET/quickdraw.zip
unzip -n -d $TARGET $TARGET/real.zip
unzip -n -d $TARGET $TARGET/sketch.zip
python scripts/split_test_domainnet.py

