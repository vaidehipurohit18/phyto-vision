/* Encyclopedia Live Search & Plant Filter JavaScript */
document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('encyclopedia-search');
    const plantFilter = document.getElementById('plant-filter');
    const grid = document.getElementById('encyclopedia-grid');
    const cards = document.querySelectorAll('.encyclopedia-card');
    const emptyCard = document.getElementById('empty-encyclopedia-card');

    if (!grid || !cards.length) return;

    function filterEncyclopedia() {
        const query = searchInput ? searchInput.value.toLowerCase().trim() : '';
        const selectedPlant = plantFilter ? plantFilter.value.toLowerCase() : 'all';

        let visibleCount = 0;

        cards.forEach(card => {
            const cardPlant = card.getAttribute('data-plant');
            const searchData = card.getAttribute('data-search');

            const matchesPlant = (selectedPlant === 'all' || cardPlant === selectedPlant);
            const matchesQuery = (!query || searchData.includes(query));

            if (matchesPlant && matchesQuery) {
                card.classList.remove('hidden');
                visibleCount++;
            } else {
                card.classList.add('hidden');
            }
        });

        if (emptyCard) {
            if (visibleCount === 0) {
                emptyCard.classList.remove('hidden');
            } else {
                emptyCard.classList.add('hidden');
            }
        }
    }

    if (searchInput) {
        searchInput.addEventListener('input', filterEncyclopedia);
    }

    if (plantFilter) {
        plantFilter.addEventListener('change', filterEncyclopedia);
    }
});
