/* History Page Table & Cards Filtering JavaScript */
document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('history-search');
    const statusFilter = document.getElementById('history-status-filter');
    const tableContainer = document.getElementById('history-table-container');
    const tableBody = document.getElementById('history-table-body');
    const cardsContainer = document.getElementById('history-cards-container');
    const emptyCard = document.getElementById('empty-history-card');
    const toggleTableBtn = document.getElementById('toggle-table-btn');
    const toggleCardsBtn = document.getElementById('toggle-cards-btn');

    if (!tableBody) return;

    let recordsData = [];
    let currentView = 'table'; // 'table' or 'cards'

    // Load History Records from API
    async function loadHistory() {
        const query = searchInput ? searchInput.value.trim() : '';
        const status = statusFilter ? statusFilter.value : 'all';

        try {
            const res = await fetch(`/api/history?search=${encodeURIComponent(query)}&status=${encodeURIComponent(status)}`);
            const data = await res.json();

            if (data.success) {
                recordsData = data.records;
                renderHistory();
            }
        } catch (err) {
            console.error('Error loading prediction history:', err);
        }
    }

    function renderHistory() {
        if (!recordsData || recordsData.length === 0) {
            tableContainer.classList.add('hidden');
            cardsContainer.classList.add('hidden');
            emptyCard.classList.remove('hidden');
            return;
        }

        emptyCard.classList.add('hidden');

        if (currentView === 'table') {
            tableContainer.classList.remove('hidden');
            cardsContainer.classList.add('hidden');
            renderTable();
        } else {
            tableContainer.classList.add('hidden');
            cardsContainer.classList.remove('hidden');
            renderCards();
        }
    }

    function renderTable() {
        tableBody.innerHTML = recordsData.map(r => `
            <tr>
                <td>
                    <img src="/${r.image_path}" alt="Leaf" class="table-thumb">
                </td>
                <td><strong>${r.plant_name}</strong></td>
                <td>${r.disease_name}</td>
                <td>
                    <span class="confidence-badge">${r.confidence}%</span>
                </td>
                <td>
                    <span class="badge ${r.status.toLowerCase() === 'healthy' ? 'badge-healthy' : 'badge-diseased'}">
                        ${r.status}
                    </span>
                </td>
                <td><small class="text-muted">${r.prediction_date}</small></td>
                <td class="text-right">
                    <a href="/result/${r.id}" class="btn btn-sm btn-secondary" title="View Diagnosis">
                        <i class="fa-solid fa-eye"></i> View
                    </a>
                    <button class="btn btn-sm btn-danger delete-btn" data-id="${r.id}" title="Delete Record">
                        <i class="fa-solid fa-trash"></i>
                    </button>
                </td>
            </tr>
        `).join('');

        attachDeleteListeners();
    }

    function renderCards() {
        cardsContainer.innerHTML = recordsData.map(r => `
            <div class="card history-card">
                <div class="history-card-header">
                    <img src="/${r.image_path}" alt="Leaf" class="history-card-img">
                    <span class="badge ${r.status.toLowerCase() === 'healthy' ? 'badge-healthy' : 'badge-diseased'}">
                        ${r.status}
                    </span>
                </div>
                <div class="history-card-body">
                    <h3>${r.plant_name} - ${r.disease_name}</h3>
                    <p class="text-muted">Confidence: <strong>${r.confidence}%</strong></p>
                    <p class="text-muted"><small>${r.prediction_date}</small></p>
                </div>
                <div class="history-card-actions">
                    <a href="/result/${r.id}" class="btn btn-sm btn-secondary">
                        <i class="fa-solid fa-eye"></i> Details
                    </a>
                    <button class="btn btn-sm btn-danger delete-btn" data-id="${r.id}">
                        <i class="fa-solid fa-trash"></i> Delete
                    </button>
                </div>
            </div>
        `).join('');

        attachDeleteListeners();
    }

    function attachDeleteListeners() {
        document.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const id = btn.getAttribute('data-id');
                if (confirm(`Are you sure you want to delete prediction record #${id}?`)) {
                    try {
                        const res = await fetch(`/api/history/${id}`, { method: 'DELETE' });
                        const data = await res.json();
                        if (data.success) {
                            loadHistory();
                        } else {
                            alert(data.error || 'Failed to delete record.');
                        }
                    } catch (err) {
                        alert('Server error deleting record.');
                    }
                }
            });
        });
    }

    // View Toggles
    if (toggleTableBtn && toggleCardsBtn) {
        toggleTableBtn.addEventListener('click', () => {
            currentView = 'table';
            toggleTableBtn.classList.add('active');
            toggleCardsBtn.classList.remove('active');
            renderHistory();
        });

        toggleCardsBtn.addEventListener('click', () => {
            currentView = 'cards';
            toggleCardsBtn.classList.add('active');
            toggleTableBtn.classList.remove('active');
            renderHistory();
        });
    }

    // Debounced Search and Filter Listener
    let debounceTimer;
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(loadHistory, 300);
        });
    }

    if (statusFilter) {
        statusFilter.addEventListener('change', loadHistory);
    }

    // Initial Load
    loadHistory();
});
