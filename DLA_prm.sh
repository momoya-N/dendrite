gcc DLA_in_el.c -o DLA
(
for i in {0.0,2.0,4.0,6.0,8.0,9.0,10.0,12.0,14.0,16.0,18.0,20.0};
do
    echo -e $i
done
) | xargs -L 1 -P 2 ./DLA