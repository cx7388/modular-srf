/**
 * Card Management Utilities
 * 
 * This module provides functionality for managing criterion and blank cards in the SRF interface,
 * including drag-and-drop operations, card creation, and grid management.
 * 
 * @module cardUtils
 */

// Counters for generating unique card IDs
let criteria_counter = 0;
let white_counter = 0;

function getMethodContextForWhiteCards() {
    const selectedMethod = document.getElementById('srf_method')?.value || '';
    const modularOptions = selectedMethod === 'modular_srf'
        && typeof collectModularOptionsFromDom === 'function'
        ? collectModularOptionsFromDom()
        : null;
    const methodForInputs = selectedMethod === 'modular_srf'
        && modularOptions
        && typeof deriveModularInputProfile === 'function'
        ? deriveModularInputProfile(modularOptions)
        : selectedMethod;
    const modularImpreciseDistance = selectedMethod === 'modular_srf'
        && modularOptions
        && (modularOptions.procedure === 'standard' || modularOptions.procedure === 'zero')
        && modularOptions.distance_type === 'imprecise';
    const modularDirect = selectedMethod === 'modular_srf'
        && modularOptions
        && modularOptions.procedure === 'direct';

    return {
        selectedMethod,
        methodForInputs,
        modularImpreciseDistance,
        modularDirect
    };
}

function shouldLockWhiteCardsForCurrentMethod() {
    const {
        selectedMethod,
        methodForInputs,
        modularImpreciseDistance,
        modularDirect
    } = getMethodContextForWhiteCards();

    if (selectedMethod === 'modular_srf') {
        return modularDirect || modularImpreciseDistance;
    }

    return new Set([
        'wap',
        'hfl_srf',
        'imprecise_srf',
        'belief_degree_imprecise_srf'
    ]).has(methodForInputs);
}


/**
 * Enables drop functionality for dragged cards.
 * @param {DragEvent} event - The drag event
 */
function allowDrop(event) {
    // prevents the cards from being dropped when they are dragged over the drop zone
    event.preventDefault();
}

/**
 * Initiates drag operation for a card.
 * @param {DragEvent} event - The drag event
 */
function drag(event) {
    // to make the cards movable when dragged around
    event.dataTransfer.setData("text", event.target.id);
    event.dataTransfer.effectAllowed = "move";
}

/**
 * Handles card drop operations in the grid.
 * 
 * Places cards in the grid according to the following rules:
 * - One card per cell
 * - Consistent card types within columns
 * - Maximum one blank card per column
 * 
 * @param {DragEvent} event - The drop event
 */
function drop(event) {
    /*
    This function enforces minimum and maximum limits set in the HTML template for the z value and required precision
    */

    // identify the card and drop zone elements
    event.preventDefault();
    var card = document.getElementById(event.dataTransfer.getData("text"));
    var dropZone = event.target.closest('.drop-zone');

    if (dropZone) {
        // calculate grid position based on mouse position relative to the grid
        var dropZoneRect = dropZone.getBoundingClientRect();
        var columnCount = (window.getComputedStyle(dropZone).getPropertyValue('grid-template-columns').match(/px/g) || []).length;
        var rowCount = (window.getComputedStyle(dropZone).getPropertyValue('grid-template-rows').match(/px/g) || []).length;

        var cellWidth = dropZoneRect.width / columnCount;
        var cellHeight = dropZoneRect.height / rowCount;

        var col = Math.floor((event.clientX - dropZoneRect.left) / cellWidth) + 1;
        var row = Math.floor((event.clientY - dropZoneRect.top) / cellHeight) + 1;

        // WAP and imprecise/fuzzy distance variants do not use visual blank cards.
        if (card.classList.contains('white') && shouldLockWhiteCardsForCurrentMethod()) {
            return;
        }

        // check if a card is being added from a stack or moved within the drop zone
        var newCard = (card.parentElement.classList.contains('stack')) ? createNewCard(card) : card;

        // ensure that a cell contains only one card and if a cell is already occupied, place the card to the row below
        while (isCellOccupied(dropZone, row, col)) {
            row++;
        }

        // ensure that the group contains only one type of card and that there are no more than one white card per group
        if (isColumnConsistent(dropZone, col, newCard)) {
            newCard.style.gridRowStart = row;
            newCard.style.gridColumnStart = col;
            dropZone.appendChild(newCard);

            updateGridState();  // perform necessary layout adjustments and UI updates
        }
    }
}


/**
 * Checks if a grid cell is already occupied.
 * 
 * @param {HTMLElement} dropZone - The drop zone element
 * @param {number} row - Target row number
 * @param {number} col - Target column number
 * @returns {boolean} True if cell is occupied, false otherwise
 */
function isCellOccupied(dropZone, row, col) {
    // this function checks if a grid cell is already occupied or not (to ensure that there is only one card per cell)
    var cell = dropZone.querySelector(`.card[style*="grid-row-start: ${row};"][style*="grid-column-start: ${col};"]`);
    return cell !== null;
}


/**
 * Validates column consistency when adding a new card.
 * 
 * Ensures that:
 * 1. All cards in a column are of the same type (criterion or blank)
 * 2. A column contains at most one blank card
 * 
 * @param {HTMLElement} dropZone - The drop zone element
 * @param {number} col - Target column number
 * @param {HTMLElement} newCard - Card being added
 * @returns {boolean} True if column remains consistent, false otherwise
 */
function isColumnConsistent(dropZone, col, newCard) {
    /*
    This function ensures that the new card that is about to be placed into a group does not violate the
    consistency of that group. A group may contain only criteria cards or only a blank card.
    */

    // this function ensures that the new card that is about to be placed into a group does not violate
    const cardsInColumn = dropZone.querySelectorAll(`.card[style*="grid-column-start: ${col};"]`);

    for (const card of cardsInColumn) {
        // the condition below also requires that there can be only one white card per group (modify as needed)
        if (card.classList.toString() !== newCard.classList.toString() || newCard.classList.contains('white')) {
            return false; // column contains mixed types
        }
    }

    return true; // column consistency is preserved
}


/**
 * Creates a new card instance from a template card.
 * 
 * For criterion cards:
 * - Assigns unique ID and increments counter
 * - Makes content editable
 * - Sets default name
 * 
 * For blank cards:
 * - Assigns unique ID
 * - Keeps content non-editable
 * 
 * @param {HTMLElement} card - Template card to clone
 * @returns {HTMLElement} New card instance with delete button
 */
function createNewCard(card) {
    /*
    This function creates (or rather clones) a card in case it is accessed from the stacks on top left
    */

    // clone the card and the pertaining information
    var newCard = card.cloneNode(true);
    newCard.addEventListener('dragstart', drag);
    newCard.contentEditable = card.classList.contains('criterion');
    if (card.classList.contains('criterion')) {
        newCard.textContent = `Criterion ${++criteria_counter}`;
        newCard.id = `${card.id}_${criteria_counter}`;
    } else {
        newCard.id = `${card.id}_${++white_counter}`;
    }

    // add a delete button to every card
    let deleteButton = Object.assign(document.createElement('button'), {
        innerHTML: '<i class="fa-solid fa-trash-can"></i>',
        title: 'Delete',
        contentEditable: false,
        onclick: deleteCard
    });

    newCard.appendChild(deleteButton);
    return newCard
}


/**
 * Removes a card from the grid and updates the layout.
 * @param {Event} event - Click event from delete button
 */
function deleteCard(event) {
    // this is a callback function of the 'Delete' button, which removes the parent card
    const card = event.target.closest('.card');
    if (!card) return;
    if (card.classList.contains('white') && shouldLockWhiteCardsForCurrentMethod()) {
        return;
    }
    card.remove();
    updateGridState();  // perform necessary layout adjustments and UI updates
}
