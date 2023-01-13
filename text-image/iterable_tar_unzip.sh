# for name in `ls -d */`;
# do;
name="image_part12/"
for i in `ls $name*.tar`;
do 
mkdir ./project/dataset/laion_chinese_cwf/${i%.tar} 
tar xvf $i -C ./project/dataset/laion_chinese_cwf/${i%.tar};
done;
# done


