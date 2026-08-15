import React, { useState, useEffect, useRef } from 'react';

const API_URL = 'http://localhost:8000';

function App() {
  const [samples, setSamples] = useState([]);
  const [selectedSampleId, setSelectedSampleId] = useState('');
  const [file, setFile] = useState(null);
  const [threshold, setThreshold] = useState(0.5);
  const [mcSamples, setMcSamples] = useState(5);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('grid');
  const [results, setResults] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  const [sceneFile, setSceneFile] = useState(null);
  const [sceneData, setSceneData] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [processLoading, setProcessLoading] = useState(false);
  const [rect, setRect] = useState(null);
  const [rectWarning, setRectWarning] = useState('');
  const [sceneResults, setSceneResults] = useState(null);
  const [sceneActiveTab, setSceneActiveTab] = useState('grid');

  const fileInputRef = useRef(null);
  const sceneFileInputRef = useRef(null);
  const canvasRef = useRef(null);
  const sceneImgRef = useRef(null);
  const isDragging = useRef(false);
  const startPos = useRef({ x: 0, y: 0 });

  useEffect(() => {
    fetch(`${API_URL}/api/samples`)
      .then(res => res.json())
      .then(data => {
        setSamples(data);
        if (data.length > 0) setSelectedSampleId(data[0].id);
      })
      .catch(err => console.error('Error fetching samples:', err));
  }, []);

  useEffect(() => {
    fetch(`${API_URL}/api/model_info`)
      .then(res => res.json())
      .then(data => setModelInfo(data))
      .catch(err => console.error('Error fetching model info:', err));
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setSelectedSampleId('');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setSelectedSampleId('');
    }
  };

  const handleDragOver = (e) => e.preventDefault();

  const selectSample = (id) => {
    setSelectedSampleId(id);
    setFile(null);
  };

  const runModel = async () => {
    setLoading(true);
    setResults(null);
    const formData = new FormData();
    formData.append('threshold', threshold);
    formData.append('mc_samples', mcSamples);
    if (file) formData.append('file', file);
    else if (selectedSampleId) formData.append('sample_id', selectedSampleId);
    else {
      alert('Please select a sample image or upload a file.');
      setLoading(false);
      return;
    }
    try {
      const res = await fetch(`${API_URL}/api/process`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error('API processing error');
      const data = await res.json();
      setResults(data);
    } catch (err) {
      console.error(err);
      alert('Failed to process image. Make sure the backend server is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleSceneFileChange = async (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    setSceneFile(f);
    setSceneData(null);
    setSceneResults(null);
    setRect(null);
    setRectWarning('');
    setUploadLoading(true);
    const fd = new FormData();
    fd.append('file', f);
    try {
      const res = await fetch(`${API_URL}/api/upload_scene`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error('Upload failed');
      const data = await res.json();
      setSceneData(data);
    } catch (err) {
      console.error(err);
      alert('Failed to upload scene.');
      setSceneFile(null);
    } finally {
      setUploadLoading(false);
    }
  };

  const getCanvasCoords = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rectBounds = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rectBounds.width;
    const scaleY = canvas.height / rectBounds.height;
    return {
      x: (e.clientX - rectBounds.left) * scaleX,
      y: (e.clientY - rectBounds.top) * scaleY
    };
  };

  const drawRect = (r) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!r) return;
    ctx.fillStyle = 'rgba(0, 255, 255, 0.25)';
    ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.strokeStyle = '#00ffff';
    ctx.lineWidth = 2;
    ctx.strokeRect(r.x, r.y, r.w, r.h);
  };

  const handleCanvasMouseDown = (e) => {
    if (!sceneData) return;
    isDragging.current = true;
    const pos = getCanvasCoords(e);
    startPos.current = pos;
    setRect(null);
    setRectWarning('');
  };

  const handleCanvasMouseMove = (e) => {
    if (!isDragging.current || !sceneData) return;
    const pos = getCanvasCoords(e);
    const x = Math.min(startPos.current.x, pos.x);
    const y = Math.min(startPos.current.y, pos.y);
    const w = Math.abs(pos.x - startPos.current.x);
    const h = Math.abs(pos.y - startPos.current.y);
    drawRect({ x, y, w, h });
    setRect({ x, y, w, h });
  };

  const handleCanvasMouseUp = () => {
    isDragging.current = false;
  };

  const fullRect = rect && sceneData ? (() => {
    const scaleX = sceneData.width / sceneData.preview_width;
    const scaleY = sceneData.height / sceneData.preview_height;
    const fullW = Math.round(rect.w * scaleX);
    const fullH = Math.round(rect.h * scaleY);
    const fullX = Math.round(rect.x * scaleX);
    const fullY = Math.round(rect.y * scaleY);
    let warning = '';
    if (fullW > sceneData.max_aoi_size || fullH > sceneData.max_aoi_size) {
      warning = `Selection too large, will be cropped to ${sceneData.max_aoi_size}px`;
    }
    return { fullX, fullY, fullW: Math.min(fullW, sceneData.max_aoi_size), fullH: Math.min(fullH, sceneData.max_aoi_size), warning };
  })() : null;

  useEffect(() => {
    if (fullRect) setRectWarning(fullRect.warning);
  }, [fullRect]);

  useEffect(() => {
    if (!sceneData) return;
    const img = sceneImgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;
    const updateCanvasSize = () => {
      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;
      drawRect(rect);
    };
    updateCanvasSize();
    window.addEventListener('resize', updateCanvasSize);
    return () => window.removeEventListener('resize', updateCanvasSize);
  }, [sceneData]);

  const processSceneRegion = async () => {
    if (!fullRect || !sceneData) return;
    setProcessLoading(true);
    setSceneResults(null);
    const fd = new FormData();
    fd.append('scene_id', sceneData.scene_id);
    fd.append('x', String(fullRect.fullX));
    fd.append('y', String(fullRect.fullY));
    fd.append('width', String(fullRect.fullW));
    fd.append('height', String(fullRect.fullH));
    fd.append('threshold', String(threshold));
    fd.append('mc_samples', String(mcSamples));
    try {
      const res = await fetch(`${API_URL}/api/process_region`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error('Process failed');
      const data = await res.json();
      setSceneResults(data);
    } catch (err) {
      console.error(err);
      alert('Failed to process region.');
    } finally {
      setProcessLoading(false);
    }
  };

  const renderMetrics = (res) => {
    if (res.metrics !== null && res.metrics !== undefined) {
      return (
        <div className="metrics-row">
          <div className="metric-card">
            <div className="metric-val" style={{ color: 'var(--accent-color)' }}>
              {res.metrics.ssim.toFixed(4)}
            </div>
            <div className="metric-name">Preserved SSIM</div>
          </div>
          <div className="metric-card">
            <div className="metric-val" style={{ color: 'var(--success-color)' }}>
              {res.metrics.psnr_db.toFixed(2)} dB
            </div>
            <div className="metric-name">Peak PSNR</div>
          </div>
          <div className="metric-card">
            <div className="metric-val" style={{ color: 'var(--warning-color)' }}>
              {res.metrics.vari_rmse.toFixed(4)}
            </div>
            <div className="metric-name">VARI RMSE</div>
          </div>
        </div>
      );
    }
    return (
      <div className="metrics-row">
        <div className="metric-card">
          <div className="metric-val" style={{ color: 'var(--warning-color)' }}>
            {res.cloud_coverage.toFixed(2)}%
          </div>
          <div className="metric-name">Cloud Coverage</div>
        </div>
        {res.ndvi_before && (
          <>
            <div className="metric-card">
              <div className="metric-val" style={{ color: 'var(--success-color)' }}>
                {res.ndvi_before.ndvi_mean.toFixed(4)}
              </div>
              <div className="metric-name">NDVI Mean</div>
            </div>
            <div className="metric-card">
              <div className="metric-val" style={{ color: 'var(--accent-color)' }}>
                {res.ndvi_before.ndvi_std.toFixed(4)}
              </div>
              <div className="metric-name">NDVI Std</div>
            </div>
          </>
        )}
        <div className="metric-card" style={{ gridColumn: '1 / -1' }}>
          <div className="metric-name" style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            {res.note || 'No ground truth available for this image — showing detection and uncertainty results only'}
          </div>
        </div>
      </div>
    );
  };

  const renderModelBanner = () => {
    if (!modelInfo) return null;
    const segLabel = modelInfo.segmentation_label || '';
    const genLabel = modelInfo.generator_label || '';
    const isSegBad = /RICE2|baseline/i.test(segLabel);
    const isGenBad = /RICE2|baseline/i.test(genLabel);
    return (
      <div style={{ padding: '0.5rem 1rem', background: 'var(--surface-color)', borderBottom: '1px solid var(--border-color)', fontSize: '0.85rem' }}>
        Segmentation: <span style={{ color: isSegBad ? '#f59e0b' : 'var(--text-primary)', fontWeight: isSegBad ? 'bold' : 'normal' }}>{segLabel}</span>
        &nbsp;·&nbsp;
        Generator: <span style={{ color: isGenBad ? '#f59e0b' : 'var(--text-primary)', fontWeight: isGenBad ? 'bold' : 'normal' }}>{genLabel}</span>
      </div>
    );
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="sidebar-header">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2V6M12 18V22M4.93 4.93L7.76 7.76M16.24 16.24L19.07 19.07M2 12H6M18 12H22M4.93 19.07L7.76 16.24M16.24 7.76L19.07 4.93" stroke="#3b82f6" strokeWidth="2.5" strokeLinecap="round"/>
          </svg>
          <h1>CloudVision-RS</h1>
        </div>

        <div className="sidebar-content">
          <div>
            <h3 className="card-title">Test Samples</h3>
            <div className="sample-list">
              {samples.map(s => (
                <div 
                  key={s.id} 
                  className={`sample-item ${selectedSampleId === s.id ? 'selected' : ''}`}
                  onClick={() => selectSample(s.id)}
                >
                  <img src={`${API_URL}${s.url}`} alt={s.name} />
                  <span>{s.name}</span>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h3 className="card-title">Custom Image</h3>
            <div 
              className="dropzone"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="dropzone-icon">⏏</div>
              <p>{file ? file.name : "Drag & Drop GeoTIFF / PNG"}</p>
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                style={{ display: 'none' }} 
                accept="image/*,.tif,.tiff"
              />
            </div>
          </div>

          <div>
            <h3 className="card-title">Parameters</h3>
            
            <div className="form-group">
              <label>Seg. Threshold</label>
              <div className="slider-container">
                <input 
                  type="range" 
                  min="0.1" 
                  max="0.9" 
                  step="0.05" 
                  value={threshold} 
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                />
                <span className="range-val">{threshold.toFixed(2)}</span>
              </div>
            </div>

            <div className="form-group">
              <label>MC Dropout Samples (Uncertainty)</label>
              <div className="slider-container">
                <input 
                  type="range" 
                  min="2" 
                  max="15" 
                  step="1" 
                  value={mcSamples} 
                  onChange={(e) => setMcSamples(parseInt(e.target.value))}
                />
                <span className="range-val">{mcSamples}</span>
              </div>
            </div>
          </div>

          <button className="btn" onClick={runModel} disabled={loading}>
            {loading ? (
              <>
                <div className="loading-spinner" />
                <span>Processing...</span>
              </>
            ) : (
              <>
                <span>▶ Run Reconstruction</span>
              </>
            )}
          </button>
        </div>
      </div>

      <div className="main-workspace">
        {renderModelBanner()}

        <div className="workspace-header">
          <h2>Interactive Workspace</h2>
          {(results || sceneResults) && (
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Cloud Coverage: <span style={{ color: 'var(--warning-color)', fontWeight: 'bold' }}>
                {((results || sceneResults).cloud_coverage).toFixed(2)}%
              </span>
            </div>
          )}
        </div>

        <div className="workspace-content">
          <div className="tabs">
            <button 
              className={`tab-btn ${activeTab === 'grid' ? 'active' : ''}`}
              onClick={() => setActiveTab('grid')}
            >
              Sample Grid
            </button>
            <button 
              className={`tab-btn ${activeTab === 'scene' ? 'active' : ''}`}
              onClick={() => setActiveTab('scene')}
            >
              Real Scene
            </button>
          </div>

          {activeTab === 'grid' && (
            <>
              {results ? (
                <>
                  <div className="result-grid">
                    <div className="card">
                      <h3 className="card-title">Cloudy Input</h3>
                      <div className="image-panel">
                        <div className="panel-label">ORIGINAL</div>
                        <img src={`${API_URL}${results.cloudy_url}`} alt="Cloudy Input" />
                      </div>
                    </div>

                    <div className="card">
                      <h3 className="card-title">Segmentation Mask</h3>
                      <div className="image-panel">
                        <div className="panel-label">BINARY MASK</div>
                        <img src={`${API_URL}${results.mask_url}`} alt="Segmentation Mask" />
                      </div>
                    </div>

                    <div className="card">
                      <h3 className="card-title">Cloud-Free Reconstruction</h3>
                      <div className="image-panel">
                        <div className="panel-label">GENERATED (cGAN)</div>
                        <img src={`${API_URL}${results.clean_url}`} alt="Cloud-free output" />
                      </div>
                    </div>

                    <div className="card">
                      <h3 className="card-title">Uncertainty Heatmap</h3>
                      <div className="image-panel">
                        <div className="panel-label">MC DROPOUT VARIANCE</div>
                        <img src={`${API_URL}${results.uncertainty_url}`} alt="Uncertainty Map" />
                      </div>
                    </div>
                  </div>

                  <div className="card">
                    <h3 className="card-title">Interactive Swipe Comparison</h3>
                    <SwipeCompare 
                      leftImg={`${API_URL}${results.cloudy_url}`} 
                      rightImg={`${API_URL}${results.clean_url}`} 
                    />
                  </div>

                  <div className="card">
                    <h3 className="card-title">Spectral Consistency Metrics</h3>
                    {renderMetrics(results)}
                  </div>
                </>
              ) : (
                <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '400px', gap: '1rem', borderStyle: 'dashed' }}>
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21 15 16 10 5 21"/>
                  </svg>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>Select a test sample on the left or upload an image to begin.</p>
                </div>
              )}
            </>
          )}

          {activeTab === 'scene' && (
            <div className="card">
              <h3 className="card-title">Real Scene Upload & AOI Selection</h3>
              {!sceneData ? (
                <div>
                  <div 
                    className="dropzone"
                    onDrop={(e) => { e.preventDefault(); }}
                    onDragOver={handleDragOver}
                    onClick={() => sceneFileInputRef.current?.click()}
                  >
                    <div className="dropzone-icon">⏏</div>
                    <p>{sceneFile ? sceneFile.name : "Drag & Drop .zip / .tif scene"}</p>
                    <input 
                      type="file" 
                      ref={sceneFileInputRef} 
                      onChange={handleSceneFileChange} 
                      style={{ display: 'none' }} 
                      accept=".zip,.tif,.tiff"
                    />
                  </div>
                  {uploadLoading && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '1rem' }}>
                      <div className="loading-spinner" />
                      <span>Uploading scene...</span>
                    </div>
                  )}
                </div>
              ) : (
                <div>
                  <div style={{ position: 'relative', display: 'inline-block' }}>
                    <img 
                      ref={sceneImgRef}
                      src={`${API_URL}${sceneData.preview_url}`} 
                      alt="Scene preview"
                      style={{ maxWidth: '100%', display: 'block' }}
                      onLoad={() => {
                        const canvas = canvasRef.current;
                        const img = sceneImgRef.current;
                        if (canvas && img) {
                          canvas.width = img.clientWidth;
                          canvas.height = img.clientHeight;
                        }
                      }}
                    />
                    <canvas
                      ref={canvasRef}
                      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', cursor: 'crosshair' }}
                      onMouseDown={handleCanvasMouseDown}
                      onMouseMove={handleCanvasMouseMove}
                      onMouseUp={handleCanvasMouseUp}
                      onMouseLeave={handleCanvasMouseUp}
                    />
                  </div>
                  <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
                    <button 
                      className="btn" 
                      onClick={processSceneRegion} 
                      disabled={!fullRect || processLoading}
                    >
                      {processLoading ? (
                        <>
                          <div className="loading-spinner" />
                          <span>Processing Region...</span>
                        </>
                      ) : (
                        <span>Process Selected Region</span>
                      )}
                    </button>
                    {rectWarning && <span style={{ color: 'var(--warning-color)', fontSize: '0.85rem' }}>{rectWarning}</span>}
                  </div>
                </div>
              )}

              {sceneResults && (
                <>
                  <div className="tabs" style={{ marginTop: '1.5rem' }}>
                    <button 
                      className={`tab-btn ${sceneActiveTab === 'grid' ? 'active' : ''}`}
                      onClick={() => setSceneActiveTab('grid')}
                    >
                      Side-by-Side Grid
                    </button>
                    <button 
                      className={`tab-btn ${sceneActiveTab === 'swipe' ? 'active' : ''}`}
                      onClick={() => setSceneActiveTab('swipe')}
                    >
                      Swipe Visualizer
                    </button>
                  </div>

                  {sceneActiveTab === 'grid' && (
                    <div className="result-grid">
                      <div className="card">
                        <h3 className="card-title">Cloudy Input</h3>
                        <div className="image-panel">
                          <div className="panel-label">ORIGINAL</div>
                          <img src={`${API_URL}${sceneResults.cloudy_url}`} alt="Cloudy Input" />
                        </div>
                      </div>

                      <div className="card">
                        <h3 className="card-title">Segmentation Mask</h3>
                        <div className="image-panel">
                          <div className="panel-label">BINARY MASK</div>
                          <img src={`${API_URL}${sceneResults.mask_url}`} alt="Segmentation Mask" />
                        </div>
                      </div>

                      <div className="card">
                        <h3 className="card-title">Cloud-Free Reconstruction</h3>
                        <div className="image-panel">
                          <div className="panel-label">GENERATED (cGAN)</div>
                          <img src={`${API_URL}${sceneResults.clean_url}`} alt="Cloud-free output" />
                        </div>
                      </div>

                      <div className="card">
                        <h3 className="card-title">Uncertainty Heatmap</h3>
                        <div className="image-panel">
                          <div className="panel-label">MC DROPOUT VARIANCE</div>
                          <img src={`${API_URL}${sceneResults.uncertainty_url}`} alt="Uncertainty Map" />
                        </div>
                      </div>
                    </div>
                  )}

                  {sceneActiveTab === 'swipe' && (
                    <div className="card">
                      <h3 className="card-title">Interactive Swipe Comparison</h3>
                      <SwipeCompare 
                        leftImg={`${API_URL}${sceneResults.cloudy_url}`} 
                        rightImg={`${API_URL}${sceneResults.clean_url}`} 
                      />
                    </div>
                  )}

                  <div className="card" style={{ marginTop: '1rem' }}>
                    <h3 className="card-title">AOI Statistics</h3>
                    <div className="metrics-row">
                      <div className="metric-card">
                        <div className="metric-val" style={{ color: 'var(--warning-color)' }}>
                          {sceneResults.cloud_coverage.toFixed(2)}%
                        </div>
                        <div className="metric-name">Cloud Coverage</div>
                      </div>
                      {sceneResults.ndvi_before && (
                        <>
                          <div className="metric-card">
                            <div className="metric-val" style={{ color: 'var(--success-color)' }}>
                              {sceneResults.ndvi_before.ndvi_mean.toFixed(4)}
                            </div>
                            <div className="metric-name">NDVI Mean</div>
                          </div>
                          <div className="metric-card">
                            <div className="metric-val" style={{ color: 'var(--accent-color)' }}>
                              {sceneResults.ndvi_before.ndvi_std.toFixed(4)}
                            </div>
                            <div className="metric-name">NDVI Std</div>
                          </div>
                        </>
                      )}
                    </div>
                    {sceneResults.note && (
                      <p style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{sceneResults.note}</p>
                    )}
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SwipeCompare({ leftImg, rightImg }) {
  const [sliderPosition, setSliderPosition] = useState(50);
  const containerRef = useRef(null);

  const handleMove = (clientX) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(percentage);
  };

  const handleMouseMove = (e) => {
    if (e.buttons === 1) handleMove(e.clientX);
  };

  const handleTouchMove = (e) => {
    if (e.touches[0]) handleMove(e.touches[0].clientX);
  };

  return (
    <div 
      className="swipe-container"
      ref={containerRef}
      onMouseMove={handleMouseMove}
      onTouchMove={handleTouchMove}
      onMouseDown={(e) => handleMove(e.clientX)}
    >
      <div className="swipe-image" style={{ backgroundImage: `url(${rightImg})` }} />
      <div 
        className="swipe-image" 
        style={{ 
          backgroundImage: `url(${leftImg})`,
          clipPath: `polygon(0 0, ${sliderPosition}% 0, ${sliderPosition}% 100%, 0 100%)`
        }}
      />
      <div className="swipe-handle" style={{ left: `${sliderPosition}%` }} />
    </div>
  );
}

export default App;
