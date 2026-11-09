/* Preserve unfinished configuration forms in this browser (no server storage). */
(function () {
    "use strict";

    const form = document.querySelector("form[data-draft-key]");
    if (!form) return;

    const key = `zero2neuro:draft:v1:${form.dataset.draftKey}`;

    function readStorage() {
        try {
            const raw = window.localStorage.getItem(key);
            if (!raw) return null;
            const parsed = JSON.parse(raw);
            return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
        } catch (_) {
            // Disabled storage, corrupt data, or privacy mode must not break forms.
            return null;
        }
    }

    function collectValues() {
        const values = {};
        for (const field of form.elements) {
            if (!field.name || field.type === "submit" || field.type === "button" || field.type === "file") {
                continue;
            }
            // Include disabled/hidden fields: users can return to another network type or split.
            if (field.type === "checkbox") {
                if (!Object.prototype.hasOwnProperty.call(values, field.name)) values[field.name] = [];
                if (field.checked) values[field.name].push(field.value);
            } else if (field.type === "radio") {
                if (field.checked) values[field.name] = field.value;
            } else if (field.tagName === "SELECT" && field.multiple) {
                values[field.name] = Array.from(field.selectedOptions, option => option.value);
            } else {
                values[field.name] = field.value;
            }
        }
        return values;
    }

    function saveDraft() {
        try {
            window.localStorage.setItem(key, JSON.stringify(collectValues()));
        } catch (_) {
            // App still works if quota is exceeded or localStorage is unavailable.
        }
    }

    function restoreDraft() {
        const saved = readStorage();
        if (!saved) return;
        for (const field of form.elements) {
            if (!field.name || !Object.prototype.hasOwnProperty.call(saved, field.name) || field.type === "file") continue;
            const value = saved[field.name];
            if (field.type === "checkbox") {
                field.checked = Array.isArray(value) && value.includes(field.value);
            } else if (field.type === "radio") {
                field.checked = value === field.value;
            } else if (field.tagName === "SELECT" && field.multiple) {
                for (const option of field.options) {
                    option.selected = Array.isArray(value) && value.includes(option.value);
                }
            } else if (typeof value === "string") {
                field.value = value;
            }
        }

        // These functions are defined by the existing per-page scripts.
        // Update conditional visibility *after* restoring the controlling inputs.
        if (form.dataset.draftKey === "network" && typeof updateNetworkType === "function") {
            updateNetworkType();
        } else if (form.dataset.draftKey === "data" && typeof syncDataForm === "function") {
            syncDataForm();
        } else if (form.dataset.draftKey === "experiment" && typeof updateEarlyStopping === "function") {
            updateEarlyStopping();
        }
    }

    // Clearing is scoped to this page's draft, never to other tabs.
    // Navigate to a fresh GET so server-rendered POST values and validation
    // messages cannot reappear when the form is cleared.
    const clearButton = form.querySelector("[data-clear-inputs]");
    if (clearButton) {
        clearButton.addEventListener("click", () => {
            if (!window.confirm("Clear all inputs on this tab? This cannot be undone.")) return;
            try {
                window.localStorage.removeItem(key);
            } catch (_) {
                // Still reset the form when browser storage is unavailable.
            }
            window.location.assign(window.location.pathname);
        });
    }

    restoreDraft();
    form.addEventListener("input", saveDraft);
    form.addEventListener("change", saveDraft);
    form.addEventListener("submit", saveDraft);
})();
