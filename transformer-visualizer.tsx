import React, { useState } from 'react';
import { Play, Info, RefreshCw } from 'lucide-react';

const TransformerVisualizer = () => {
  const [activeTab, setActiveTab] = useState('attention');
  const [animationStep, setAnimationStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  // Sample data for visualizations
  const sentence = ["The", "cat", "sat", "on", "mat"];
  
  // Simulated attention weights (Query: "cat" attending to all tokens)
  const attentionWeights = [
    [0.1, 0.05, 0.05, 0.05, 0.05], // The
    [0.15, 0.6, 0.1, 0.05, 0.1],   // cat (high self-attention)
    [0.1, 0.2, 0.5, 0.1, 0.1],     // sat
    [0.05, 0.1, 0.15, 0.5, 0.2],   // on
    [0.05, 0.05, 0.1, 0.2, 0.6]    // mat
  ];

  const startAnimation = () => {
    setIsAnimating(true);
    setAnimationStep(0);
    let step = 0;
    const interval = setInterval(() => {
      step++;
      if (step > 4) {
        clearInterval(interval);
        setIsAnimating(false);
        step = 0;
      }
      setAnimationStep(step);
    }, 1000);
  };

  const SelfAttentionViz = () => (
    <div className="space-y-6">
      <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
        <div className="flex items-start gap-2">
          <Info className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
          <div className="text-sm text-blue-900">
            <p className="font-semibold mb-1">Self-Attention Mechanism</p>
            <p>Each word attends to every word in the sequence (including itself). Darker lines indicate stronger attention weights.</p>
          </div>
        </div>
      </div>

      <div className="flex justify-center mb-4">
        <button
          onClick={startAnimation}
          disabled={isAnimating}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
        >
          {isAnimating ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          {isAnimating ? 'Animating...' : 'Animate Attention'}
        </button>
      </div>

      <div className="relative bg-white p-8 rounded-lg border-2 border-gray-200">
        {/* Source tokens */}
        <div className="flex justify-around mb-20">
          {sentence.map((word, i) => (
            <div
              key={`src-${i}`}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                isAnimating && i === animationStep
                  ? 'bg-blue-600 text-white scale-110 shadow-lg'
                  : 'bg-gray-100 text-gray-700'
              }`}
            >
              {word}
            </div>
          ))}
        </div>

        {/* Attention lines */}
        <svg className="absolute top-0 left-0 w-full h-full pointer-events-none" style={{ zIndex: 1 }}>
          {sentence.map((_, queryIdx) => 
            sentence.map((_, keyIdx) => {
              const weight = attentionWeights[queryIdx][keyIdx];
              const x1 = ((queryIdx + 0.5) / sentence.length) * 100;
              const y1 = 20;
              const x2 = ((keyIdx + 0.5) / sentence.length) * 100;
              const y2 = 80;
              
              const shouldShow = !isAnimating || queryIdx === animationStep;
              const opacity = shouldShow ? weight : 0;
              
              return (
                <line
                  key={`line-${queryIdx}-${keyIdx}`}
                  x1={`${x1}%`}
                  y1={`${y1}%`}
                  x2={`${x2}%`}
                  y2={`${y2}%`}
                  stroke={queryIdx === keyIdx ? '#3b82f6' : '#6366f1'}
                  strokeWidth={weight * 4 + 1}
                  opacity={opacity}
                  className="transition-opacity duration-300"
                />
              );
            })
          )}
        </svg>

        {/* Target tokens */}
        <div className="flex justify-around mt-20">
          {sentence.map((word, i) => (
            <div
              key={`tgt-${i}`}
              className="px-4 py-2 bg-purple-100 text-purple-700 rounded-lg font-medium"
            >
              {word}
            </div>
          ))}
        </div>
      </div>

      {/* Attention matrix */}
      <div className="bg-white p-6 rounded-lg border-2 border-gray-200">
        <h3 className="text-lg font-semibold mb-4">Attention Weight Matrix</h3>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="border p-2 bg-gray-50"></th>
                {sentence.map((word, i) => (
                  <th key={i} className="border p-2 bg-purple-50 text-purple-700 font-semibold">
                    {word}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sentence.map((word, i) => (
                <tr key={i}>
                  <td className="border p-2 bg-blue-50 text-blue-700 font-semibold">{word}</td>
                  {attentionWeights[i].map((weight, j) => (
                    <td
                      key={j}
                      className="border p-2 text-center transition-all"
                      style={{
                        backgroundColor: `rgba(59, 130, 246, ${weight})`,
                        color: weight > 0.4 ? 'white' : 'black',
                        fontWeight: isAnimating && i === animationStep ? 'bold' : 'normal'
                      }}
                    >
                      {weight.toFixed(2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-sm text-gray-600 mt-3">
          Rows = Query tokens, Columns = Key tokens. Each cell shows attention weight.
        </p>
      </div>
    </div>
  );

  const PositionalEncodingViz = () => {
    const positions = [0, 1, 2, 3, 4, 5, 6, 7];
    const dimensions = 8;
    
    // Simplified positional encoding calculation
    const getPositionalEncoding = (pos, i, d) => {
      const angle = pos / Math.pow(10000, (2 * i) / d);
      return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    };

    return (
      <div className="space-y-6">
        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
          <div className="flex items-start gap-2">
            <Info className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
            <div className="text-sm text-green-900">
              <p className="font-semibold mb-1">Positional Encoding</p>
              <p>Sine and cosine functions at different frequencies encode position information. Each position has a unique pattern across dimensions.</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg border-2 border-gray-200">
          <h3 className="text-lg font-semibold mb-4">Positional Encoding Heatmap</h3>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <th className="border p-2 bg-gray-50 text-sm">Position</th>
                  {[...Array(dimensions)].map((_, i) => (
                    <th key={i} className="border p-2 bg-gray-50 text-sm">
                      d{i}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {positions.map((pos) => (
                  <tr key={pos}>
                    <td className="border p-2 bg-blue-50 font-semibold text-center">{pos}</td>
                    {[...Array(dimensions)].map((_, i) => {
                      const value = getPositionalEncoding(pos, i, dimensions);
                      const normalized = (value + 1) / 2; // Normalize to 0-1
                      return (
                        <td
                          key={i}
                          className="border p-1"
                          style={{
                            backgroundColor: `rgba(${value > 0 ? '59, 130, 246' : '239, 68, 68'}, ${Math.abs(value)})`,
                            width: '60px',
                            height: '40px'
                          }}
                          title={value.toFixed(3)}
                        />
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-sm text-gray-600 mt-3">
            Blue = positive values (sine/cosine positive), Red = negative values. Intensity shows magnitude.
          </p>
        </div>

        <div className="bg-white p-6 rounded-lg border-2 border-gray-200">
          <h3 className="text-lg font-semibold mb-4">PE Formula</h3>
          <div className="bg-gray-50 p-4 rounded font-mono text-sm space-y-2">
            <div>PE(pos, 2i) = sin(pos / 10000^(2i/d_model))</div>
            <div>PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))</div>
          </div>
          <p className="text-sm text-gray-600 mt-3">
            Where pos = position, i = dimension index, d_model = embedding dimension
          </p>
        </div>
      </div>
    );
  };

  const MultiHeadAttentionViz = () => (
    <div className="space-y-6">
      <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
        <div className="flex items-start gap-2">
          <Info className="w-5 h-5 text-purple-600 mt-0.5 flex-shrink-0" />
          <div className="text-sm text-purple-900">
            <p className="font-semibold mb-1">Multi-Head Attention</p>
            <p>Multiple attention mechanisms run in parallel, each learning different aspects of the relationships between tokens.</p>
          </div>
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg border-2 border-gray-200">
        <h3 className="text-lg font-semibold mb-4">8 Attention Heads</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[...Array(8)].map((_, headIdx) => (
            <div key={headIdx} className="border-2 border-purple-200 rounded-lg p-3">
              <div className="text-center font-semibold mb-2 text-purple-700">
                Head {headIdx + 1}
              </div>
              <div className="grid grid-cols-5 gap-1">
                {sentence.map((_, i) => (
                  <div
                    key={i}
                    className="aspect-square rounded"
                    style={{
                      backgroundColor: `hsl(${(headIdx * 45 + i * 20) % 360}, 70%, 60%)`,
                      opacity: 0.6 + Math.random() * 0.4
                    }}
                    title={`Head ${headIdx + 1}, Token ${i}`}
                  />
                ))}
              </div>
              <div className="text-xs text-gray-500 mt-2 text-center">
                Focus: {['Syntax', 'Semantics', 'Position', 'Relations', 'Context', 'Grammar', 'Meaning', 'Structure'][headIdx]}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg border-2 border-gray-200">
        <h3 className="text-lg font-semibold mb-4">Architecture Flow</h3>
        <div className="flex flex-col items-center space-y-4">
          <div className="w-full max-w-md bg-blue-100 p-4 rounded-lg text-center font-semibold">
            Input Embeddings (d_model = 512)
          </div>
          <div className="text-2xl text-gray-400">↓</div>
          <div className="w-full max-w-md border-2 border-purple-300 rounded-lg p-4">
            <div className="text-center font-semibold mb-3">Split into 8 heads (64 dims each)</div>
            <div className="grid grid-cols-4 gap-2">
              {[...Array(8)].map((_, i) => (
                <div key={i} className="bg-purple-100 p-2 rounded text-center text-sm">
                  H{i+1}
                </div>
              ))}
            </div>
          </div>
          <div className="text-2xl text-gray-400">↓</div>
          <div className="w-full max-w-md bg-purple-100 p-4 rounded-lg text-center">
            Parallel Attention Computation
          </div>
          <div className="text-2xl text-gray-400">↓</div>
          <div className="w-full max-w-md bg-green-100 p-4 rounded-lg text-center font-semibold">
            Concatenate & Linear (back to d_model = 512)
          </div>
        </div>
      </div>
    </div>
  );

  const ArchitectureViz = () => (
    <div className="space-y-6">
      <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
        <div className="flex items-start gap-2">
          <Info className="w-5 h-5 text-orange-600 mt-0.5 flex-shrink-0" />
          <div className="text-sm text-orange-900">
            <p className="font-semibold mb-1">Complete Transformer Architecture</p>
            <p>The full encoder-decoder structure with all components working together for sequence-to-sequence tasks.</p>
          </div>
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg border-2 border-gray-200">
        <div className="flex flex-col lg:flex-row gap-8 items-start justify-center">
          {/* Encoder */}
          <div className="flex-1 max-w-sm">
            <h3 className="text-xl font-bold text-center mb-4 text-blue-700">ENCODER</h3>
            <div className="space-y-3">
              <div className="bg-blue-100 p-3 rounded-lg text-center font-semibold">
                Input Embedding
              </div>
              <div className="bg-blue-100 p-3 rounded-lg text-center">
                + Positional Encoding
              </div>
              
              <div className="border-2 border-blue-400 rounded-lg p-4 space-y-3">
                <div className="text-center font-semibold text-blue-700 mb-2">
                  Encoder Layer (×N)
                </div>
                <div className="bg-purple-100 p-3 rounded text-center">
                  Multi-Head<br/>Self-Attention
                </div>
                <div className="text-center text-gray-500">↓ Add & Norm</div>
                <div className="bg-green-100 p-3 rounded text-center">
                  Feed Forward<br/>Network
                </div>
                <div className="text-center text-gray-500">↓ Add & Norm</div>
              </div>
              
              <div className="bg-blue-200 p-3 rounded-lg text-center font-semibold">
                Encoder Output
              </div>
            </div>
          </div>

          {/* Connection */}
          <div className="hidden lg:flex items-center justify-center">
            <div className="text-4xl text-gray-400">→</div>
          </div>

          {/* Decoder */}
          <div className="flex-1 max-w-sm">
            <h3 className="text-xl font-bold text-center mb-4 text-purple-700">DECODER</h3>
            <div className="space-y-3">
              <div className="bg-purple-100 p-3 rounded-lg text-center font-semibold">
                Output Embedding
              </div>
              <div className="bg-purple-100 p-3 rounded-lg text-center">
                + Positional Encoding
              </div>
              
              <div className="border-2 border-purple-400 rounded-lg p-4 space-y-3">
                <div className="text-center font-semibold text-purple-700 mb-2">
                  Decoder Layer (×N)
                </div>
                <div className="bg-purple-100 p-3 rounded text-center">
                  Masked Multi-Head<br/>Self-Attention
                </div>
                <div className="text-center text-gray-500">↓ Add & Norm</div>
                <div className="bg-indigo-100 p-3 rounded text-center">
                  Cross-Attention<br/>(Encoder-Decoder)
                </div>
                <div className="text-center text-gray-500">↓ Add & Norm</div>
                <div className="bg-green-100 p-3 rounded text-center">
                  Feed Forward<br/>Network
                </div>
                <div className="text-center text-gray-500">↓ Add & Norm</div>
              </div>
              
              <div className="bg-purple-200 p-3 rounded-lg text-center font-semibold">
                Linear & Softmax
              </div>
              <div className="bg-purple-300 p-3 rounded-lg text-center font-bold">
                Output Probabilities
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg border-2 border-gray-200">
        <h3 className="text-lg font-semibold mb-3">Key Components</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="border border-gray-200 rounded p-3">
            <div className="font-semibold text-blue-700 mb-1">Residual Connections</div>
            <p className="text-sm text-gray-600">Add input to output of each sublayer (Add & Norm)</p>
          </div>
          <div className="border border-gray-200 rounded p-3">
            <div className="font-semibold text-purple-700 mb-1">Layer Normalization</div>
            <p className="text-sm text-gray-600">Normalizes activations across features</p>
          </div>
          <div className="border border-gray-200 rounded p-3">
            <div className="font-semibold text-green-700 mb-1">Feed Forward Network</div>
            <p className="text-sm text-gray-600">Two linear layers with ReLU: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂</p>
          </div>
          <div className="border border-gray-200 rounded p-3">
            <div className="font-semibold text-indigo-700 mb-1">Masked Attention</div>
            <p className="text-sm text-gray-600">Prevents attending to future positions during training</p>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Transformer Architecture Visualizer
          </h1>
          <p className="text-gray-600">
            Interactive visualization of key Transformer components
          </p>
        </div>

        {/* Tabs */}
        <div className="flex flex-wrap gap-2 mb-6 bg-white p-2 rounded-lg shadow-md">
          {[
            { id: 'attention', label: 'Self-Attention', icon: '🎯' },
            { id: 'positional', label: 'Positional Encoding', icon: '📍' },
            { id: 'multihead', label: 'Multi-Head Attention', icon: '🧠' },
            { id: 'architecture', label: 'Full Architecture', icon: '🏗️' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 min-w-[140px] px-4 py-3 rounded-lg font-medium transition-all ${
                activeTab === tab.id
                  ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="transition-all duration-300">
          {activeTab === 'attention' && <SelfAttentionViz />}
          {activeTab === 'positional' && <PositionalEncodingViz />}
          {activeTab === 'multihead' && <MultiHeadAttentionViz />}
          {activeTab === 'architecture' && <ArchitectureViz />}
        </div>
      </div>
    </div>
  );
};

export default TransformerVisualizer;