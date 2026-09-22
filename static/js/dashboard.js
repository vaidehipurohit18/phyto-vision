/* Dashboard Analytics Chart.js JavaScript */
document.addEventListener('DOMContentLoaded', async () => {
    const dashboardContent = document.getElementById('dashboard-content');
    const emptyCard = document.getElementById('empty-dashboard-card');

    if (!dashboardContent) return;

    try {
        const response = await fetch('/api/dashboard-stats');
        const stats = await response.json();

        if (stats.total_predictions === 0) {
            dashboardContent.classList.add('hidden');
            emptyCard.classList.remove('hidden');
            return;
        }

        // Update Stat Counters
        document.getElementById('stat-total').textContent = stats.total_predictions;
        document.getElementById('stat-healthy').textContent = stats.healthy_count;
        document.getElementById('stat-diseased').textContent = stats.diseased_count;
        document.getElementById('stat-confidence').textContent = `${stats.avg_confidence}%`;

        // 1. Ratio Chart (Doughnut)
        const ctxRatio = document.getElementById('ratioChart').getContext('2d');
        new Chart(ctxRatio, {
            type: 'doughnut',
            data: {
                labels: ['Healthy Leaves', 'Diseased Leaves'],
                datasets: [{
                    data: [stats.healthy_count, stats.diseased_count],
                    backgroundColor: ['#10b981', '#ef4444'],
                    borderWidth: 2,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });

        // 2. Top Diseases Chart (Bar)
        const diseaseLabels = stats.top_diseases.map(item => item.disease);
        const diseaseCounts = stats.top_diseases.map(item => item.count);

        const ctxDiseases = document.getElementById('diseasesChart').getContext('2d');
        new Chart(ctxDiseases, {
            type: 'bar',
            data: {
                labels: diseaseLabels.length > 0 ? diseaseLabels : ['No Diseases Recorded'],
                datasets: [{
                    label: 'Detection Frequency',
                    data: diseaseCounts.length > 0 ? diseaseCounts : [0],
                    backgroundColor: '#1b4d3e',
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: { beginAtZero: true, ticks: { stepSize: 1 } }
                }
            }
        });

        // 3. Trend Line Chart
        const trendDates = stats.recent_trend.map(item => item.date);
        const trendCounts = stats.recent_trend.map(item => item.count);
        const trendConf = stats.recent_trend.map(item => item.avg_conf);

        const ctxTrend = document.getElementById('trendChart').getContext('2d');
        new Chart(ctxTrend, {
            type: 'line',
            data: {
                labels: trendDates.length > 0 ? trendDates : ['Today'],
                datasets: [
                    {
                        label: 'Daily Scans Count',
                        data: trendCounts.length > 0 ? trendCounts : [0],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true,
                        tension: 0.3,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Avg Confidence (%)',
                        data: trendConf.length > 0 ? trendConf : [0],
                        borderColor: '#3b82f6',
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.3,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { type: 'linear', display: true, position: 'left', beginAtZero: true },
                    y1: { type: 'linear', display: true, position: 'right', min: 0, max: 100 }
                }
            }
        });

    } catch (err) {
        console.error('Failed fetching dashboard analytics stats:', err);
    }
});
