/**
 * Main Application Logic
 * 
 * This module handles the core functionality of the SRF web application, including:
 * - Card arrangement management (import/export)
 * - Grid layout and column management
 * - User input validation
 * - Event listeners for main UI interactions
 * 
 * @module main
 */

document.addEventListener('DOMContentLoaded', () => {
    updateGridState();  // perform necessary layout adjustments and UI updates
});


/**
 * Clear button click handler - Removes all cards from the drop zone
 * and resets the layout width.
 */
document.querySelector('.clear-button').addEventListener('click', () => {
    // Remove all cards from the drop zone
    document.querySelectorAll('.drop-zone .card').forEach(card => {
        card.remove();
    });

    // Reset drop zone width to default
    document.querySelector(".drop-zone").style.minWidth = '100%';
    document.querySelector(".ranks-container").style.minWidth = '100%';

    updateGridState();  // perform necessary layout adjustments and UI updates
});


/**
 * Collects current configurable form values by input/select id.
 */
function collectConfigFormValues() {
    const selectors = [
        '#additional-inputs input[id]',
        '#additional-inputs select[id]',
        '#optional-constraints-panel input[id]',
        '#optional-constraints-panel select[id]'
    ];

    const formValues = {};
    document.querySelectorAll(selectors.join(',')).forEach(el => {
        if (!el.id) return;
        if (el.type === 'checkbox') {
            formValues[el.id] = Boolean(el.checked);
        } else {
            formValues[el.id] = el.value;
        }
    });
    return formValues;
}


/**
 * Ensures dynamic belief-row inputs exist before applying imported values.
 */
function ensureDynamicInputExists(inputId) {
    const zMatch = /^z-(value|beta)-(\d+)$/.exec(inputId);
    if (zMatch) {
        const targetIndex = parseInt(zMatch[2], 10);
        while (!document.getElementById(`z-value-${targetIndex}`) && typeof addBeliefRow === 'function') {
            addBeliefRow('z');
        }
        return;
    }

    const eMatch = /^e-(value|beta)-(\d+)-(\d+)$/.exec(inputId);
    if (eMatch) {
        const rank = parseInt(eMatch[2], 10);
        const targetIndex = parseInt(eMatch[3], 10);
        while (!document.getElementById(`e-value-${rank}-${targetIndex}`) && typeof addBeliefRow === 'function') {
            addBeliefRow('e', rank);
            if (!document.getElementById(`e-value-${rank}-1`)) break;
        }
    }
}


function applyConfigFormEntries(entries) {
    entries.forEach(([id, value]) => {
        ensureDynamicInputExists(id);
        const el = document.getElementById(id);
        if (!el) return;
        if (el.type === 'checkbox') {
            el.checked = Boolean(value);
        } else {
            el.value = value;
        }
    });
}


/**
 * Applies imported form values to existing controls.
 */
function applyConfigFormValues(formValues) {
    if (!formValues || typeof formValues !== 'object') return;

    const entries = Object.entries(formValues);
    const structuralIds = new Set([
        'mod-q3-procedure',
        'mod-q4-distance',
        'mod-q5-distance-format',
        'mod-q6-z',
        'mod-q7-z-format',
        'mod-q8-prob',
        'mod-q11-output',
        'mod-q12-unit',
        'mod-q13-var-method'
    ]);

    const structuralEntries = entries.filter(([id]) => structuralIds.has(id));
    const remainingEntries = entries.filter(([id]) => !structuralIds.has(id));

    if (structuralEntries.length > 0) {
        applyConfigFormEntries(structuralEntries);
        if (typeof updateGridState === 'function') {
            updateGridState();
        }
    }

    applyConfigFormEntries(remainingEntries);

    if (typeof syncOptionalConstraintPanels === 'function') {
        syncOptionalConstraintPanels();
    }
    if (typeof validateMethodInputsAndToggleRun === 'function') {
        validateMethodInputsAndToggleRun();
    }
}


/**
 * Exports a full elicitation snapshot (cards + method inputs).
 */
document.querySelector('.export-button').addEventListener('click', () => {
    const dropZone = document.querySelector('.drop-zone');
    const cards_arrangement = Array.from(dropZone.children).map((card) => ({
        id: card.id,
        name: card.textContent.split('Delete')[0].trim(),
        class: card.classList.contains('criterion') ? 'criterion' : 'white',
        col: card.style.gridColumnStart,
        row: card.style.gridRowStart
    }));

    const srfMethod = document.getElementById('srf_method')?.value || 'srf';
    const snapshot = {
        format: 'srf-elicitation-config-v2',
        exported_at: new Date().toISOString(),
        srf_method: srfMethod,
        cards_arrangement,
        form_values: collectConfigFormValues()
    };

    const blob = new Blob([JSON.stringify(snapshot, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'srf_elicitation_config.json';
    link.click();
});


/**
 * Import button click handler - Loads a card arrangement from a JSON file.
 */
document.querySelector('.import-button').addEventListener('click', () => {
    const fileInput = document.getElementById("importFile");
    fileInput.value = '';  // Clear previous selection
    fileInput.click();

    fileInput.onchange = (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = function (e) {
            try {
                const data = JSON.parse(e.target.result);

                // Backward-compatible import: plain card array.
                if (Array.isArray(data)) {
                    restoreCards(data);
                    updateGridState();
                    return;
                }

                if (!data || typeof data !== 'object') {
                    throw new Error('Invalid import file format.');
                }

                const arrangement = Array.isArray(data.cards_arrangement) ? data.cards_arrangement : [];
                if (arrangement.length) {
                    restoreCards(arrangement);
                }
                updateGridState();

                const methodSelect = document.getElementById('srf_method');
                const importedMethod = data.srf_method;
                if (methodSelect && typeof importedMethod === 'string') {
                    const methodExists = Array.from(methodSelect.options).some(opt => opt.value === importedMethod);
                    if (methodExists) {
                        methodSelect.value = importedMethod;
                        if (typeof updateSimosMethodInfo === 'function') {
                            updateSimosMethodInfo(importedMethod);
                        } else {
                            updateGridState();
                        }
                    }
                }

                applyConfigFormValues(data.form_values || {});
            } catch (error) {
                console.error('Import error:', error);
                alert('Invalid import file. Please select a valid SRF configuration JSON.');
            }
        };
        reader.readAsText(file);
    }
});

/**
 * Restores a card arrangement from imported data.
 * 
 * @param {Array} arrangement - Array of card objects with position and content data
 */
function restoreCards(arrangement) {
    const dropZone = document.querySelector('.drop-zone');
    dropZone.innerHTML = ''; // Clear existing cards

    arrangement.forEach(card => {
        var newCard = createNewCard(document.querySelector(`.card.${card.class}`))

        newCard.id = card.id;
        newCard.textContent = card.name;
        newCard.style.gridRowStart = card.row;
        newCard.style.gridColumnStart = card.col;

        // Add delete button to card
        var deleteButton = Object.assign(document.createElement('button'), {
            innerHTML: '<i class="fa-solid fa-trash-can"></i>',
            title: 'Delete',
            contentEditable: false,
            onclick: deleteCard
        });
        newCard.appendChild(deleteButton);

        dropZone.appendChild(newCard);
    });
}


// Initialize collapsible buttons
document.querySelectorAll(".button-collapsible").forEach(button => {
    button.addEventListener("click", function () {
        this.classList.toggle("active");
        let content = this.nextElementSibling;
        content.style.maxHeight = content.style.maxHeight ? null : content.scrollHeight + "px";
    });
});


/**
 * Grid Layout Management
 * 
 * Handles the dynamic grid layout including:
 * - Column insertion
 * - Gap positioning
 * - Insert button placement
 */
document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.querySelector(".drop-zone");
    const ranksContainer = document.querySelector(".ranks-container");
    const insertButton = document.querySelector(".insert-column-btn");

    const computedStyle = window.getComputedStyle(dropZone);
    const columnWidth = parseInt(computedStyle.getPropertyValue("grid-template-columns").split(" ")[0], 10);
    const gap = parseInt(computedStyle.getPropertyValue("gap"), 10) || 0;

    /**
     * Calculates positions of gaps between grid columns.
     * @returns {Array<number>} Array of x-coordinates for column gaps
     */
    function getGapPositions() {
        const columns = Math.floor(dropZone.offsetWidth / (columnWidth + gap));
        const paddingLeft = parseFloat(computedStyle.getPropertyValue('padding-left'));
        const gapPositions = [];

        // calculate the positions of the gaps between grid cells
        for (let i = 1; i < columns; i++) {
            gapPositions.push(paddingLeft + i * (columnWidth + gap) - gap / 2);
        }

        return gapPositions;
    }

    /**
     * Positions the insert column button near the closest gap.
     * @param {MouseEvent} event - Mouse movement event
     */
    function positionInsertButton(event) {
        const gapPositions = getGapPositions();
        const mouseX = event.clientX
                             - document.querySelector("main").offsetLeft
                             + document.querySelector(".drop-zone-container").scrollLeft; // Adjust to container reference

        // Find closest gap position
        const closestGap = gapPositions.reduce((prev, curr) =>
            Math.abs(curr - mouseX) < Math.abs(prev - mouseX) ? curr : prev
        );
        insertButton.style.left = `${closestGap - insertButton.offsetWidth / 2}px`;

        if (dropZone.children.length === 0) {
            insertButton.style.opacity = "0";
            return;
        }

        const cards = Array.from(dropZone.children);
        const cardPositions = cards.map(card => card.offsetLeft);

        // Find the closest card position
        const closestCard = cardPositions.reduce((prev, curr) =>
            Math.abs(curr - mouseX) < Math.abs(prev - mouseX) ? curr : prev
        );

        const isWithinThreshold = Math.abs(closestCard + columnWidth / 2 - mouseX) <= columnWidth;
        insertButton.style.opacity = isWithinThreshold ? "1" : "0";
    }

    // Add a new column to the grid
    insertButton.addEventListener("click", () => {
        let cards = Array.from(dropZone.children);

        // Sort cards by offsetLeft in descending order (right to left)
        let sortedCards = cards.sort((a, b) => b.offsetLeft - a.offsetLeft);
        let cardPositions = sortedCards.map(card => card.offsetLeft);

        const insertPos = parseFloat(insertButton.style.left);

        // Shift cards right of insertion point
        for (let i = 0; i < sortedCards.length; i++) {
            if (cardPositions[i] > insertPos) {
                sortedCards[i].style.gridColumnStart = parseInt(sortedCards[i].style.gridColumnStart) + 1;
            }
        }

        // Update grid width
        const widthIncrement = (columnWidth + gap) / document.querySelector(".drop-zone-container").offsetWidth * 100;
        let currentMinWidth = parseFloat(dropZone.style.minWidth) || (100 + widthIncrement);
        dropZone.style.minWidth = `${currentMinWidth + widthIncrement}%`;
        ranksContainer.style.minWidth = dropZone.style.minWidth;

        updateGridState();  // perform necessary layout adjustments and UI updates
    });

    // Track mouse for insert button positioning
    dropZone.addEventListener("mouseenter", positionInsertButton);
    dropZone.addEventListener("mousemove", positionInsertButton);
});


/**
 * Enforces min/max limits on numeric input fields.
 * 
 * @param {Event} event - Input change event
 */
function enforceMinMaxLimits(event) {
    let { max, min, value, step } = event.target;
    value = Math.min(Math.max(parseFloat(value), parseFloat(min)), parseFloat(max));
    
    // Convert to integer if step is 1
    event.target.value = step === 1 ? parseInt(value) : value;
}
