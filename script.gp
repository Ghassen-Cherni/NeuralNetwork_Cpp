set title 'Income vs Happiness'
set xlabel 'Income'
set ylabel 'Happiness'
plot 'data.txt' using 1:2 with points title 'Predicted Happiness', 'data.txt' using 1:3 with points title 'Ground Truth Happiness', 'data.txt' using 1:2 with lines title 'Predicted Line'
