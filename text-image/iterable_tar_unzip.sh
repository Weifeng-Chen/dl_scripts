# for name in `ls -d */`;
# do;
name="image_part12/"
for i in `ls $name*.tar`;
do 
mkdir /cognitive_comp/chenweifeng/project/dataset/laion_chinese_cwf/${i%.tar} 
tar xvf $i -C /cognitive_comp/chenweifeng/project/dataset/laion_chinese_cwf/${i%.tar};
done;
# done


# /shared_space/ccnl/mm_data/laion2B-multi-chinese-subset/