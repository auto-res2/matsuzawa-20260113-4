import re

def fix_math_expressions(text):
    # Replace mathematical formulas in text with proper LaTeX
    
    # Pattern 1: L_reg = \lambda_ctgr ∑_l ||γ_l^{\tau} − μ_l||^2
    text = re.sub(
        r'L_reg = \\lambda_ctgr ∑_l \|\|γ_l\^\{\\tau\} − μ_l\|\|',
        r'$L_{\\text{reg}} = \\lambda_{\\text{ctgr}} \\sum_l \\|\\gamma_l^{\\tau} - \\mu_l\\|',
        text
    )
    
    # Pattern 2: μ_l ← (1−η) μ_l + η · mean_{\tau} (γ_l^{\tau})
    text = re.sub(
        r'μ_l ← \(1−η\) μ_l \+ η · mean_\{\\tau\} \(γ_l\^\{\\tau\}\)',
        r'$\\mu_l \\leftarrow (1-\\eta) \\mu_l + \\eta \\cdot \\text{mean}_{\\tau} (\\gamma_l^{\\tau})$',
        text
    )
    
    # Pattern 3: Δ_l = ∑_{\tau'} S_{\tau,\tau'} γ_l^{\tau'}
    text = re.sub(
        r'Δ_l = ∑_\{\\tau\'\} S_\{\\tau,\\tau\'\} γ_l\^\{\\tau\'\}',
        r'$\\Delta_l = \\sum_{\\tau\'} S_{\\tau,\\tau\'} \\gamma_l^{\\tau\'}$',
        text
    )
    
    # Pattern 4: μ_l ← (1−η) μ_l + η Δ_l
    text = re.sub(
        r'μ_l ← \(1−η\) μ_l \+ η Δ_l',
        r'$\\mu_l \\leftarrow (1-\\eta) \\mu_l + \\eta \\Delta_l$',
        text
    )
    
    # Pattern 5: Simple inline τ that's already in math mode
    text = re.sub(r'τ in the batch', r'$\\tau$ in the batch', text)
    
    # Pattern 6: {μ_l} with μ_l ∈
    text = re.sub(
        r'\{μ_l\} with μ_l ∈',
        r'{$\\mu_l$} with $\\mu_l \\in$',
        text
    )
    
    # Pattern 7: λ_ctgr and similar
    text = re.sub(r'λ_ctgr([,\s])', r'$\\lambda_{\\text{ctgr}}$\\1', text)
    text = re.sub(r'η ∈', r'$\\eta \\in$', text)
    
    # Pattern 8: S_{τ,τ'} ≥ 0 and ∑_{τ'} S_{τ,τ'} = 1
    text = re.sub(
        r'S_\{τ,τ\'\} ≥ 0 and ∑_\{τ\'\} S_\{τ,τ\'\} = 1',
        r'$S_{\\tau,\\tau\'} \\geq 0$ and $\\sum_{\\tau\'} S_{\\tau,\\tau\'} = 1$',
        text
    )
    
    # Pattern 9: L_total = ∑_{τ} ...
    text = re.sub(
        r'L_total = ∑_\{τ\} L_task\(τ\) \+ L_reg\(τ\)',
        r'$L_{\\text{total}} = \\sum_{\\tau} L_{\\text{task}}(\\tau) + L_{\\text{reg}}(\\tau)$',
        text
    )
    
    # Pattern 10: L_reg(τ) = λ_ctgr ∑_l ||γ_l^{τ} − μ_l||^2
    text = re.sub(
        r'L_reg\(τ\) = λ_ctgr ∑_l \|\|γ_l\^\{τ\} − μ_l\|\|',
        r'$L_{\\text{reg}}(\\tau) = \\lambda_{\\text{ctgr}} \\sum_l \\|\\gamma_l^{\\tau} - \\mu_l\\|',
        text
    )
    
    # Pattern 11: γ_l^{τ} for the reg-term
    text = re.sub(r'γ_l\^\{τ\} for the reg-term', r'$\\gamma_l^{\\tau}$ for the reg-term', text)
    
    # Pattern 12: σ_l^2(τ)
    text = re.sub(r'σ_l\^2\(τ\)', r'$\\sigma_l^2(\\tau)$', text)
    
    # Pattern 13: η in hyperparameters
    text = re.sub(r'η in \{', r'$\\eta$ in {', text)
    
    return text

# Read the file
with open('main.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Apply fixes
content = fix_math_expressions(content)

# Write back
with open('main.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed Unicode characters in mathematical expressions")
