gcc DLA_in_el.c -o DLA
(
for i in {1.0,2.0,3.0} #Voltage
#for i in {0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1} #sticking probability
do
  #for j in 0.0
  for j in {0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0} #alpha
  do
    echo -e $i $j
  done
done
    
) | xargs -L 1 -P 2 ./DLA