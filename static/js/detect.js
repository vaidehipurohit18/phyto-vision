/* Leaf Image Detection Upload JavaScript */
document.addEventListener('DOMContentLoaded', () => {
    const detectForm = document.getElementById('detect-form');
    const fileInput = document.getElementById('leaf_image');
    const dropZone = document.getElementById('drop-zone');
    const dropPrompt = document.getElementById('drop-zone-prompt');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('image-preview');
    const removeImgBtn = document.getElementById('remove-img-btn');
    const errorBox = document.getElementById('detect-error-box');
    const errorMsg = document.getElementById('detect-error-msg');
    const submitBtn = document.getElementById('submit-detect-btn');
    const loadingOverlay = document.getElementById('loading-overlay');

    if (!detectForm) return;

    // Trigger file dialog on dropzone click
    dropZone.addEventListener('click', (e) => {
        if (e.target.closest('#remove-img-btn')) return;
        fileInput.click();
    });

    // Drag & Drop handlers
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            fileInput.files = files;
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    function handleFileSelect(file) {
        hideError();

        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            showError(`Invalid file format '${file.type}'. Please upload JPG, PNG, or WEBP image.`);
            resetFile();
            return;
        }

        if (file.size > 16 * 1024 * 1024) {
            showError(`File size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds maximum 16 MB limit.`);
            resetFile();
            return;
        }

        // Show image preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            dropPrompt.classList.add('hidden');
            previewContainer.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    removeImgBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetFile();
    });

    function resetFile() {
        fileInput.value = '';
        previewImage.src = '#';
        previewContainer.classList.add('hidden');
        dropPrompt.classList.remove('hidden');
        hideError();
    }

    function showError(msg) {
        errorMsg.textContent = msg;
        errorBox.classList.remove('hidden');
    }

    function hideError() {
        errorBox.classList.add('hidden');
        errorMsg.textContent = '';
    }

    // Submit Form via AJAX
    detectForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideError();

        if (!fileInput.files || fileInput.files.length === 0) {
            showError('Please select or capture a plant leaf image first.');
            return;
        }

        const formData = new FormData();
        formData.append('leaf_image', fileInput.files[0]);

        // Show loading spinner
        loadingOverlay.classList.remove('hidden');
        submitBtn.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok && data.success) {
                // Redirect to result page
                window.location.href = data.redirect_url;
            } else {
                loadingOverlay.classList.add('hidden');
                submitBtn.disabled = false;
                showError(data.error || 'An error occurred during leaf analysis.');
            }
        } catch (err) {
            loadingOverlay.classList.add('hidden');
            submitBtn.disabled = false;
            showError('Network error or server connection failed. Please try again.');
        }
    });
});
