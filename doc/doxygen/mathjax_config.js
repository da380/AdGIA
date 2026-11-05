MathJax.Hub.Config({
    TeX: {
        Macros: {
            // Differential d for integrals
            dd: '\\,\\mathrm{d}', // This is for \dd

            // Vectors and matrices
            bvec: ['{\\mathbf{#1}}', 1],
            bmat: ['{\\mathbf{#1}}', 1],

            // Operators
            grad: '\\nabla',
            divg: '\\nabla \\cdot',
            curl: '\\nabla \\times',
            deriv: '\\mathrm{D}',

            // Sets
            realset: '\\mathbb{R}',

            // Norms and absolute values
            norm: ['\\left\\|#1\\right\\|', 1],
            abs: ['\\left|#1\\right|', 1],

            // FEM specific
            jump: ['\\left[\\!\\left[#1\\right]\\!\\right]', 1],
            avg: ['\\left\\{\\!\\left\\{#1\\right\\}\\!\\right\\}', 1],
        }
    }
});