import './style.css'
import { Chart, registerables } from 'chart.js'
Chart.register(...registerables)

let masterData = null
let charts = {
    asr: null,
    latency: null,
    evasion: null
}

async function init() {
    try {
        const response = await fetch('/data.json')
        masterData = await response.json()
        
        setupFilters()
        renderDashboard()
    } catch (error) {
        console.error('Error loading dashboard data:', error)
    }
}

function setupFilters() {
    const modelFilter = document.getElementById('model-filter')
    const templateFilter = document.getElementById('template-filter')
    const compressorFilter = document.getElementById('compressor-filter')

    masterData.models.forEach(model => {
        const opt = document.createElement('option')
        opt.value = model
        opt.textContent = model.split('/').pop()
        modelFilter.appendChild(opt)
    })

    masterData.templates.forEach(template => {
        const opt = document.createElement('option')
        opt.value = template
        opt.textContent = template.charAt(0).toUpperCase() + template.slice(1)
        templateFilter.appendChild(opt)
    })

    if (masterData.compressors) {
        masterData.compressors.forEach(compressor => {
            const opt = document.createElement('option')
            opt.value = compressor
            opt.textContent = compressor.charAt(0).toUpperCase() + compressor.slice(1)
            compressorFilter.appendChild(opt)
        })
    }

    modelFilter.addEventListener('change', renderDashboard)
    templateFilter.addEventListener('change', renderDashboard)
    compressorFilter.addEventListener('change', renderDashboard)
}

function getFilteredData() {
    const model = document.getElementById('model-filter').value
    const template = document.getElementById('template-filter').value
    const compressor = document.getElementById('compressor-filter').value

    return masterData.entries.filter(entry => {
        const modelMatch = (model === 'all' || entry.model === model)
        const templateMatch = (template === 'all' || entry.template === template)
        const compressorMatch = (compressor === 'all' || entry.compressor === compressor)
        return modelMatch && templateMatch && compressorMatch
    })
}

function renderDashboard() {
    const data = getFilteredData()
    renderStats(data)
    renderASRChart(data)
    renderLatencyChart(data)
    renderEvasionChart(data)
}

function renderStats(data) {
    const totalTrials = data.reduce((acc, curr) => acc + curr.total_samples, 0)
    const avgAsrSin = data.reduce((acc, curr) => acc + curr.asr_sin, 0) / (data.length || 1)
    const avgAsrCon = data.reduce((acc, curr) => acc + curr.asr_con, 0) / (data.length || 1)
    const avgRatio = data.reduce((acc, curr) => acc + curr.ratio, 0) / (data.length || 1)

    document.getElementById('total-trials').textContent = totalTrials.toLocaleString()
    document.getElementById('avg-asr-original').textContent = `${(avgAsrSin * 100).toFixed(1)}%`
    document.getElementById('avg-asr-compressed').textContent = `${(avgAsrCon * 100).toFixed(1)}%`
    document.getElementById('avg-ratio').textContent = `${(avgRatio * 100).toFixed(1)}%`
}

function renderASRChart(data) {
    const ctx = document.getElementById('asr-chart').getContext('2d')
    if (charts.asr) charts.asr.destroy()

    // Group by Model + Template + Compressor
    const groups = {}
    const groupBaseline = {} // asr_sin per group for baseline point at 0%
    data.forEach(entry => {
        const compString = entry.compressor ? `[${entry.compressor}]` : ''
        const key = `${entry.model.split('/').pop()} (${entry.template}) ${compString}`
        if (!groups[key]) groups[key] = []
        const compressionPct = Math.round((1 - entry.rate) * 100)
        const actualPct = Math.round((1 - entry.ratio) * 100)
        groups[key].push({ x: compressionPct, y: entry.asr_con * 100, real: actualPct, isAbstractive: entry.compressor === 'abstractive' })
        // Accumulate baseline ASR (asr_sin is the same for all rates in the same experiment)
        if (!groupBaseline[key]) groupBaseline[key] = []
        groupBaseline[key].push(entry.asr_sin * 100)
    })

    // Prepend baseline point at 0% compression (sin compresión)
    Object.keys(groups).forEach(key => {
        const avgBaseline = groupBaseline[key].reduce((a, b) => a + b, 0) / groupBaseline[key].length
        groups[key].unshift({ x: 0, y: avgBaseline })
    })

    const datasets = Object.keys(groups).map((key, index) => ({
        label: key,
        data: groups[key].sort((a, b) => a.x - b.x), // Low to high compression
        borderColor: getChartColor(index),
        backgroundColor: getChartColor(index, 0.2),
        tension: 0.3,
        fill: false,
        pointRadius: 6,
        borderWidth: 3
    }))

    charts.asr = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(1) + '% ASR';
                            }
                            if (context.raw && context.raw.isAbstractive) {
                                label += ` (Nivel Real Logrado: ${context.raw.real}%)`;
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Compression Level (%)', color: '#94a3b8' },
                    min: 0,
                    max: 100,
                    ticks: { callback: (v) => v + '%', color: '#94a3b8', stepSize: 10 },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                },
                y: {
                    beginAtZero: true,
                    title: { display: true, text: 'Attack Success Rate (%)', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                }
            },
            plugins: {
                legend: { labels: { color: '#f8fafc', font: { family: 'Outfit' } } }
            }
        }
    })
}

function renderLatencyChart(data) {
    const ctx = document.getElementById('latency-chart').getContext('2d')
    if (charts.latency) charts.latency.destroy()

    // Aggregate by template
    const stats_template = {}
    data.forEach(entry => {
        if (!stats_template[entry.template]) stats_template[entry.template] = { sin: 0, con: 0, count: 0 }
        stats_template[entry.template].sin += entry.latency_sin
        stats_template[entry.template].con += (entry.latency_con + entry.latency_comp)
        stats_template[entry.template].count++
    })

    const labels = Object.keys(stats_template)
    const sin_data = labels.map(l => stats_template[l].sin / stats_template[l].count)
    const con_data = labels.map(l => stats_template[l].con / stats_template[l].count)

    charts.latency = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels.map(l => l.toUpperCase()),
            datasets: [
                { label: 'Original (sec)', data: sin_data, backgroundColor: '#6366f1' },
                { label: 'Compressed + Overhead (sec)', data: con_data, backgroundColor: '#f43f5e' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#f8fafc' } } },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                x: { ticks: { color: '#94a3b8' } }
            }
        }
    })
}

function renderEvasionChart(data) {
    const ctx = document.getElementById('evasion-chart').getContext('2d')
    if (charts.evasion) charts.evasion.destroy()

    // Histogram of tokens
    charts.evasion = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['ASR', 'Compression Efficiency', 'Latency Overhead', 'Evasion Rate', 'Robustness'],
            datasets: [{
                label: 'Global Performance Area',
                data: [65, 80, 40, 90, 70], // Sample radar data for visualization
                fill: true,
                backgroundColor: 'rgba(16, 185, 129, 0.2)',
                borderColor: '#10b981',
                pointBackgroundColor: '#10b981',
                pointBorderColor: '#fff',
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255,255,255,0.1)' },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    pointLabels: { color: '#94a3b8' },
                    ticks: { display: false }
                }
            }
        }
    })
}

const colors = ['#6366f1', '#f43f5e', '#10b981', '#38bdf8', '#fbbf24', '#a855f7']
function getChartColor(i, alpha = 1) {
    const base = colors[i % colors.length]
    if (alpha === 1) return base
    return base + Math.round(alpha * 255).toString(16).padStart(2, '0')
}

window.onload = init
