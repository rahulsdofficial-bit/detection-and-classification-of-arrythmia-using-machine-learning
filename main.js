// Arrhythmia Detection System Frontend Helpers

document.addEventListener('DOMContentLoaded', function () {
    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(function (alert) {
        setTimeout(function () {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });

    // Sample selector helper on analysis page
    const sampleSelect = document.getElementById('sampleSelect');
    const ecgFileInput = document.getElementById('ecg_file');
    const sampleNotice = document.getElementById('sampleNotice');

    if (sampleSelect && ecgFileInput) {
        sampleSelect.addEventListener('change', function () {
            if (this.value) {
                ecgFileInput.value = '';
                if (sampleNotice) {
                    sampleNotice.style.display = 'block';
                    sampleNotice.textContent = 'Selected demo sample: ' + this.options[this.selectedIndex].text;
                }
            } else {
                if (sampleNotice) {
                    sampleNotice.style.display = 'none';
                }
            }
        });

        ecgFileInput.addEventListener('change', function () {
            if (this.files.length > 0) {
                sampleSelect.value = '';
                if (sampleNotice) {
                    sampleNotice.style.display = 'none';
                }
            }
        });
    }

    // Submit button loading spinner
    const analyzeForm = document.getElementById('analysisForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeForm && analyzeBtn) {
        analyzeForm.addEventListener('submit', function () {
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Analyzing ECG Signals...';
        });
    }
});
