<h1>QutritLab</h1>



QutritLab is a Python library for simulating ternary based quantum circuits.



Features:

  - Single-qutrit gates (H, X, Y, Z, RX, RY, RZ)

  - Multi-qutrit gates (CSUM, CCX, CCSUM)

  - Custom gates

  - Ability to apply gates to specific subspaces

  - Controlled operations with arbitrary control states

  - Circuit abstraction



<h2>Example</h2>

<!-- HTML generated using hilite.me --><div style="background: #272822; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%;"><span></span><span style="color: #FF4689">from</span><span style="color: #F8F8F2"> qutritlab </span><span style="color: #FF4689">import</span> <span style="color: #FF4689">*</span>

<span style="color: #F8F8F2">c</span> <span style="color: #FF4689">=</span> <span style="color: #F8F8F2">Circuit(num_qutrits</span><span style="color: #FF4689">=</span><span style="color: #AE81FF">2</span><span style="color: #F8F8F2">)</span> <span style="color: #959077"># Initialise 2 qutrit circuit</span>

<span style="color: #F8F8F2">c</span><span style="color: #FF4689">.</span><span style="color: #F8F8F2">add_operation(gate</span><span style="color: #FF4689">=</span><span style="color: #F8F8F2">H(),</span> <span style="color: #F8F8F2">targets</span><span style="color: #FF4689">=</span><span style="color: #F8F8F2">[</span><span style="color: #AE81FF">0</span><span style="color: #F8F8F2">])</span> <span style="color: #959077"># Apply Hadamard gate to q0</span>
<span style="color: #F8F8F2">c</span><span style="color: #FF4689">.</span><span style="color: #F8F8F2">add_operation(gate</span><span style="color: #FF4689">=</span><span style="color: #F8F8F2">CSUM(),</span> <span style="color: #F8F8F2">controls</span><span style="color: #FF4689">=</span><span style="color: #F8F8F2">[</span><span style="color: #AE81FF">0</span><span style="color: #F8F8F2">],</span> <span style="color: #F8F8F2">targets</span><span style="color: #FF4689">=</span><span style="color: #F8F8F2">[</span><span style="color: #AE81FF">1</span><span style="color: #F8F8F2">])</span> <span style="color: #959077"># Apply CSUM gate to q1</span>

<span style="color: #F8F8F2">results</span> <span style="color: #FF4689">=</span> <span style="color: #F8F8F2">Simulator()</span><span style="color: #FF4689">.</span><span style="color: #F8F8F2">run(c,</span> <span style="color: #F8F8F2">shots</span><span style="color: #FF4689">=</span><span style="color: #AE81FF">1000</span><span style="color: #F8F8F2">,</span> <span style="color: #F8F8F2">ternary_string</span><span style="color: #FF4689">=</span><span style="color: #66D9EF">True</span><span style="color: #F8F8F2">)</span> <span style="color: #959077"># Obtain results</span>

<span style="color: #F8F8F2">print(</span><span style="color: #E6DB74">&quot;State:&quot;</span><span style="color: #F8F8F2">,</span> <span style="color: #F8F8F2">results</span><span style="color: #FF4689">.</span><span style="color: #F8F8F2">state)</span>
<span style="color: #F8F8F2">print(</span><span style="color: #E6DB74">&quot;Probabilities:&quot;</span><span style="color: #F8F8F2">,</span> <span style="color: #F8F8F2">results</span><span style="color: #FF4689">.</span><span style="color: #F8F8F2">probabilities)</span>
<span style="color: #F8F8F2">print(</span><span style="color: #E6DB74">&quot;Measurements:&quot;</span><span style="color: #F8F8F2">,</span> <span style="color: #F8F8F2">results</span><span style="color: #FF4689">.</span><span style="color: #F8F8F2">measurements)</span>
</pre><br></div>


