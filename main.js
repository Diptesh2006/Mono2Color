/**
 * Frontend JavaScript for AI Image Colorization Web App
 * Handles file upload, API calls, and result display
 */

const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const uploadArea = document.getElementById('uploadArea');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const removeBtn = document.getElementById('removeBtn');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const originalImage = document.getElementById('originalImage');
const colorizedImage = document.getElementById('colorizedImage');
const downloadBtn = document.getElementById('downloadBtn');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');

let selectedFile = null;

// Event Listeners
browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
removeBtn.addEventListener('click', removeFile);

// Drag and drop handlers
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);
uploadArea.addEventListener('click', () => fileInput.click());

// Download button handler
downloadBtn.addEventListener('click', downloadColorizedImage);

/**
 * Handle file selection from input
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

/**
 * Handle drag over event
 */
function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

/**
 * Handle drag leave event
 */
function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

/**
 * Handle drop event
 */
function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            processFile(file);
        } else {
            showError('Please drop an image file');
        }
    }
}

/**
 * Process selected file
 */
function processFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please select an image file (JPEG, PNG, GIF, or BMP)');
        return;
    }

    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        showError('File size too large. Please select an image smaller than 10MB');
        return;
    }

    selectedFile = file;
    
    // Display file info
    fileName.textContent = file.name;
    fileInfo.style.display = 'flex';
    
    // Hide previous results and errors
    hideResults();
    hideError();

    // Display preview of original image
    const reader = new FileReader();
    reader.onload = (e) => {
        originalImage.src = e.target.result;
        // Automatically start colorization
        colorizeImage(file);
    };
    reader.readAsDataURL(file);
}

/**
 * Remove selected file
 */
function removeFile() {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.style.display = 'none';
    hideResults();
    hideError();
    originalImage.src = '';
}

/**
 * Colorize image via API
 */
async function colorizeImage(file) {
    // Show loading state
    showLoading();
    hideError();
    hideResults();

    // Prepare form data
    const formData = new FormData();
    formData.append('image', file);

    try {
        // Send POST request to colorize endpoint
        const response = await fetch('/colorize', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            // Display results
            colorizedImage.src = data.colorized;
            showResults();
        } else {
            showError(data.error || 'Colorization failed. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Network error. Please check your connection and try again.');
    } finally {
        hideLoading();
    }
}

/**
 * Show loading state
 */
function showLoading() {
    loadingSection.style.display = 'block';
}

/**
 * Hide loading state
 */
function hideLoading() {
    loadingSection.style.display = 'none';
}

/**
 * Show results
 */
function showResults() {
    resultsSection.style.display = 'block';
}

/**
 * Hide results
 */
function hideResults() {
    resultsSection.style.display = 'none';
}

/**
 * Show error message
 */
function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'block';
}

/**
 * Hide error message
 */
function hideError() {
    errorMessage.style.display = 'none';
}

/**
 * Download colorized image
 */
function downloadColorizedImage() {
    if (!colorizedImage.src) {
        showError('No colorized image available');
        return;
    }

    // Create a temporary anchor element to trigger download
    const link = document.createElement('a');
    link.href = colorizedImage.src;
    link.download = `colorized_${selectedFile ? selectedFile.name : 'image.jpg'}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

