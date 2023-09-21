gcc DLA_in_el.c -o DLA -lm #以下MaxPhi,alpha,numの順で記述
(
#for i in 1.0 #sticking probability
for i in {1.0,0.7,0.4,0.1} #sticking probability
do
  #for j in 0.1
  for j in {0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0} #alpha
  do
    for k in {0..20} #data index
    do
      echo -e $i $j $k
    done
  done
done
    
) | xargs -L 1 -P 11 ./DLA