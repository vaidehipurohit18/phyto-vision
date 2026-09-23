/* ============================================================
   PHYTO VISION - LEAF IMAGE DETECTION
   Mobile Camera + Gallery + Desktop File Upload
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {

    const detectForm = document.getElementById('detect-form');

    const cameraInput = document.getElementById('camera-input');
    const galleryInput = document.getElementById('gallery-input');

    const cameraBtn = document.getElementById('camera-btn');
    const galleryBtn = document.getElementById('gallery-btn');

    const dropZone = document.getElementById('drop-zone');
    const dropPrompt = document.getElementById('drop-zone-prompt');

    const previewContainer =
        document.getElementById('preview-container');

    const previewImage =
        document.getElementById('image-preview');

    const removeImgBtn =
        document.getElementById('remove-img-btn');

    const errorBox =
        document.getElementById('detect-error-box');

    const errorMsg =
        document.getElementById('detect-error-msg');

    const submitBtn =
        document.getElementById('submit-detect-btn');

    const loadingOverlay =
        document.getElementById('loading-overlay');


    if (!detectForm) {
        return;
    }


    /* ========================================================
       STORE SELECTED FILE
       ======================================================== */

    let selectedFile = null;


    /* ========================================================
       CAMERA BUTTON
       ======================================================== */

    cameraBtn.addEventListener('click', (e) => {

        e.preventDefault();
        e.stopPropagation();

        hideError();

        /*
         * This input has:
         *
         * accept="image/*"
         * capture="environment"
         *
         * On supported mobile browsers this requests
         * the rear-facing camera.
         */

        cameraInput.click();

    });


    /* ========================================================
       GALLERY BUTTON
       ======================================================== */

    galleryBtn.addEventListener('click', (e) => {

        e.preventDefault();
        e.stopPropagation();

        hideError();

        galleryInput.click();

    });


    /* ========================================================
       CAMERA INPUT CHANGE
       ======================================================== */

    cameraInput.addEventListener('change', () => {

        if (
            cameraInput.files &&
            cameraInput.files.length > 0
        ) {

            const file = cameraInput.files[0];

            handleFileSelect(file);

        }

    });


    /* ========================================================
       GALLERY INPUT CHANGE
       ======================================================== */

    galleryInput.addEventListener('change', () => {

        if (
            galleryInput.files &&
            galleryInput.files.length > 0
        ) {

            const file = galleryInput.files[0];

            handleFileSelect(file);

        }

    });


    /* ========================================================
       DESKTOP DROPZONE CLICK
       ======================================================== */

    dropZone.addEventListener('click', (e) => {

        /*
         * Don't trigger file picker if user clicked
         * one of the buttons.
         */

        if (
            e.target.closest('#camera-btn') ||
            e.target.closest('#gallery-btn') ||
            e.target.closest('#remove-img-btn')
        ) {

            return;

        }

        /*
         * On desktop, clicking the main area opens
         * the gallery/file picker.
         */

        galleryInput.click();

    });


    /* ========================================================
       DRAG & DROP
       ======================================================== */

    ['dragenter', 'dragover'].forEach(eventName => {

        dropZone.addEventListener(eventName, (e) => {

            e.preventDefault();
            e.stopPropagation();

            dropZone.classList.add('dragover');

        });

    });


    ['dragleave', 'drop'].forEach(eventName => {

        dropZone.addEventListener(eventName, (e) => {

            e.preventDefault();
            e.stopPropagation();

            dropZone.classList.remove('dragover');

        });

    });


    dropZone.addEventListener('drop', (e) => {

        const files = e.dataTransfer.files;

        if (
            files &&
            files.length > 0
        ) {

            handleFileSelect(files[0]);

        }

    });


    /* ========================================================
       FILE VALIDATION
       ======================================================== */

    function handleFileSelect(file) {

        hideError();

        if (!file) {
            return;
        }


        console.log(
            '[DETECT] Selected file:',
            file.name
        );

        console.log(
            '[DETECT] MIME type:',
            file.type
        );

        console.log(
            '[DETECT] Size:',
            file.size
        );


        const validTypes = [
            'image/png',
            'image/jpeg',
            'image/jpg',
            'image/webp'
        ];


        /*
         * Some mobile cameras may return an empty
         * MIME type even though the file is an image.
         *
         * Therefore don't reject purely because
         * file.type is empty.
         */

        if (
            file.type &&
            !validTypes.includes(file.type)
        ) {

            showError(
                `Invalid image format "${file.type}". ` +
                `Please use JPG, PNG, or WEBP.`
            );

            resetFile();

            return;

        }


        /* 16 MB limit */

        if (
            file.size >
            16 * 1024 * 1024
        ) {

            showError(
                `File size is ` +
                `${(file.size / (1024 * 1024)).toFixed(1)} MB. ` +
                `Maximum allowed size is 16 MB.`
            );

            resetFile();

            return;

        }


        selectedFile = file;


        /* ====================================================
           PREVIEW
           ==================================================== */

        const reader = new FileReader();


        reader.onload = (e) => {

            previewImage.src =
                e.target.result;

            dropPrompt.classList.add('hidden');

            previewContainer.classList.remove('hidden');

            submitBtn.disabled = false;

        };


        reader.onerror = () => {

            showError(
                'Unable to read the selected image.'
            );

        };


        reader.readAsDataURL(file);

    }


    /* ========================================================
       REMOVE IMAGE
       ======================================================== */

    removeImgBtn.addEventListener('click', (e) => {

        e.preventDefault();
        e.stopPropagation();

        resetFile();

    });


    function resetFile() {

        selectedFile = null;

        cameraInput.value = '';

        galleryInput.value = '';

        previewImage.src = '#';

        previewContainer.classList.add('hidden');

        dropPrompt.classList.remove('hidden');

        hideError();

    }


    /* ========================================================
       ERROR HANDLING
       ======================================================== */

    function showError(message) {

        errorMsg.textContent = message;

        errorBox.classList.remove('hidden');

    }


    function hideError() {

        errorBox.classList.add('hidden');

        errorMsg.textContent = '';

    }


    /* ========================================================
       SUBMIT PREDICTION
       ======================================================== */

    detectForm.addEventListener(
        'submit',
        async (e) => {

            e.preventDefault();

            hideError();


            if (!selectedFile) {

                showError(
                    'Please take a photo or choose a leaf image first.'
                );

                return;

            }


            console.log(
                '[DETECT] Sending image to /predict...'
            );


            const formData = new FormData();

            formData.append(
                'leaf_image',
                selectedFile
            );


            /* Loading UI */

            loadingOverlay.classList.remove('hidden');

            submitBtn.disabled = true;


            try {

                const response = await fetch(
                    '/predict',
                    {
                        method: 'POST',
                        body: formData
                    }
                );


                console.log(
                    '[DETECT] Server response:',
                    response.status
                );


                const data =
                    await response.json();


                if (
                    response.ok &&
                    data.success
                ) {

                    console.log(
                        '[DETECT] Prediction successful.'
                    );


                    window.location.href =
                        data.redirect_url;


                } else {

                    loadingOverlay.classList.add(
                        'hidden'
                    );

                    submitBtn.disabled = false;


                    showError(
                        data.error ||
                        'An error occurred during leaf analysis.'
                    );

                }


            } catch (err) {

                console.error(
                    '[DETECT] Request failed:',
                    err
                );


                loadingOverlay.classList.add(
                    'hidden'
                );

                submitBtn.disabled = false;


                showError(
                    'Network error or server connection failed. Please try again.'
                );

            }

        }
    );

});