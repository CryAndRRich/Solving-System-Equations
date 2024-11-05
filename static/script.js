function createMatrix() {
    const elements = document.querySelectorAll('#equation');
    elements.forEach(element => {
        element.parentNode.removeChild(element);
    });

    const input = document.getElementById('create-text');
    const matrix = document.getElementById('system-of-equations');
    if (isNaN(input.value) || Number(input.value) < 2 || Number(input.value) > 10) {
        const error = document.createElement('p');
        error.innerHTML = 'Please type in a number between 2 and 10!';
        error.id = 'equation';
        error.className = 'error';
        matrix.appendChild(error);
    }
    else {
        for (let i = 0; i < Number(input.value); i++) {
            const row = document.createElement('section');
            row.id = 'equation';
            for (let j = 1; j <= Number(input.value) + 1; j++) {
                const entry = document.createElement('input');
                entry.type = "text";
                entry.id = "entry";
                entry.name = "entry" + (j + i * (Number(input.value) + 1));
                row.appendChild(entry);
                if (j <= Number(input.value)) {
                    const text = document.createElement('p');
                    text.innerText = 'x' + j + (j < Number(input.value) ? '  +': '  =');
                    text.id = 'variable';
                    row.appendChild(text);
                }
            }
            matrix.appendChild(row);
        }
    }
}

function Submit() {
    const elements = document.querySelectorAll('#equation');
    if (elements.length < 2) {
        elements.forEach(element => {
            element.parentNode.removeChild(element);
        });
        const matrix = document.getElementById('system-of-equations');
        const error = document.createElement('p');
        error.innerHTML = 'Please create a system of equations first!';
        error.id = 'equation';
        error.className = 'error';
        matrix.appendChild(error);
    }
    else {
        const errors = document.querySelectorAll('.error');
        errors.forEach(element => {
            element.parentNode.removeChild(element);
        });
        
        if (!checkValid()) {
            const matrix = document.getElementById('create-equations');
            const error = document.createElement('p');
            error.innerHTML = "Can't apply this method! Choose another method or Create a new system of equations!";
            error.id = 'equation';
            error.className = 'error';
            matrix.appendChild(error);
        }
        else {
            document.getElementById('create-equations').submit();
        }
    }
}

function checkValid() {
    const input = document.getElementById('create-text');
    const entries = document.querySelectorAll('#entry');
    let valid = true;
    for (let i = 0; i < Number(input.value); i++) {
        const entry = entries[i + i * (Number(input.value) + 1)];
        if (isNaN(Number(entry.value)) || Number(entry.value) === 0) {
            valid = false;
            break;
        }
    }
    return valid;
}
