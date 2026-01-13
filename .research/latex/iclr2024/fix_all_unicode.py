import re

def escape_special_chars(text):
    """Escape special regex characters"""
    for char in r'()[]{}.*+?|^$\\':
        text = text.replace(char, '\\' + char)
    return text

def fix_all_unicode(content):
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        # Skip if line starts with % (comment)
        if line.strip().startswith('%'):
            continue
            
        # Replace Unicode mathematical symbols with LaTeX equivalents
        # Do this carefully to avoid breaking existing LaTeX
        
        # First, handle complete mathematical expressions
        # Pattern: things that look like variables with subscripts/superscripts
        
        # gamma with subscript/superscript
        line = re.sub(r'γ_([a-z0-9]+)\^\{([^}]+)\}', r'$\\gamma_\\1^{\\2}$', line)
        line = re.sub(r'γ_([a-z0-9]+)', r'$\\gamma_\\1$', line)
        line = re.sub(r'γ', r'$\\gamma$', line)
        
        # mu with subscript/superscript  
        line = re.sub(r'μ_([a-z0-9]+)\^\{([^}]+)\}', r'$\\mu_\\1^{\\2}$', line)
        line = re.sub(r'μ_([a-z0-9]+)', r'$\\mu_\\1$', line)
        line = re.sub(r'μ', r'$\\mu$', line)
        
        # lambda
        line = re.sub(r'λ_([a-z0-9]+)', r'$\\lambda_\\1$', line)
        line = re.sub(r'λ', r'$\\lambda$', line)
        
        # eta
        line = re.sub(r'η', r'$\\eta$', line)
        
        # sum
        line = re.sub(r'∑_([a-z0-9]+)', r'$\\sum_\\1$', line)
        line = re.sub(r'∑_\{([^}]+)\}', r'$\\sum_{\\1}$', line)
        line = re.sub(r'∑', r'$\\sum$', line)
        
        # tau
        line = re.sub(r'τ', r'$\\tau$', line)
        
        # minus sign
        line = re.sub(r'−', r'-', line)
        
        # Delta
        line = re.sub(r'Δ_([a-z0-9]+)', r'$\\Delta_\\1$', line)
        line = re.sub(r'Δ', r'$\\Delta$', line)
        
        # sigma
        line = re.sub(r'σ_([a-z0-9]+)\^([0-9]+)\(([^)]+)\)', r'$\\sigma_\\1^\\2(\\3)$', line)
        line = re.sub(r'σ', r'$\\sigma$', line)
        
        # left arrow
        line = re.sub(r'←', r'$\\leftarrow$', line)
        
        # geq, times
        line = re.sub(r'≥', r'$\\geq$', line)
        line = re.sub(r'×', r'$\\times$', line)
        
        # Clean up double math modes ($$...$$ -> $...$)
        while '$$' in line:
            line = line.replace('$$', '$')
        
        # Clean up empty math modes
        line = re.sub(r'\$\s*\$', '', line)
        
        lines[i] = line
    
    return '\n'.join(lines)

# Read file
with open('main.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix Unicode
content = fix_all_unicode(content)

# Write back
with open('main.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed all Unicode mathematical symbols")
