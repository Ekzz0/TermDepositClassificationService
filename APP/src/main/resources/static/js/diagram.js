const ctx = document.getElementById('myChart');
const dates = ["15.1.2000", "16.1.2000", "17.1.2000", "20.1.2000"]
const probability = [10, 20, 40, 50];

chart =  new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [{
        label: '% вероятность ',
        data: probability,
        borderColor: '#36A2EB',
        backgroundColor: '#9BD0F5',
        borderWidth: 1
      }]
    },
    options: {
      scales: {
        y: {
          beginAtZero: false
        }
      }
    }
});
console.log(chart)