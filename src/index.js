const HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SEC-EDGAR-GPT</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #1a1a1a; line-height: 1.6; }
  .container { max-width: 720px; margin: 0 auto; padding: 2rem 1.5rem; }
  h1 { font-size: 2rem; margin-bottom: 0.5rem; }
  .subtitle { color: #555; font-size: 1.1rem; margin-bottom: 2rem; }
  h2 { font-size: 1.3rem; margin: 2rem 0 0.8rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
  p { margin-bottom: 1rem; }
  a { color: #0066cc; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .links { display: flex; gap: 1.5rem; margin-bottom: 2rem; flex-wrap: wrap; }
  .links a { background: #f5f5f5; padding: 0.6rem 1.2rem; border-radius: 6px; font-weight: 500; }
  .links a:hover { background: #e8e8e8; text-decoration: none; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 1rem; }
  th, td { text-align: left; padding: 0.5rem 0.8rem; border-bottom: 1px solid #eee; }
  th { font-weight: 600; }
  .key { color: #555; width: 200px; }
  ul { padding-left: 1.5rem; margin-bottom: 1rem; }
  li { margin-bottom: 0.3rem; }
  .footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd; color: #888; font-size: 0.9rem; }
</style>
</head>
<body>
<div class="container">
<h1>SEC-EDGAR-GPT</h1>
<p class="subtitle">A 124M parameter GPT-2 trained on SEC-EDGAR financial filings</p>
<div class="links">
  <a href="https://github.com/lzwjava/sec-edgar-gpt">GitHub</a>
  <a href="https://huggingface.co/lzwjava/sec-edgar-gpt-124m-hf">Model (HuggingFace)</a>
  <a href="https://github.com/lzwjava/sec-edgar-gpt/raw/main/sec-edgar-gpt.pdf">Paper (PDF)</a>
</div>
<h2>Overview</h2>
<p>A 124M parameter GPT-2 language model trained on 1.55 billion tokens of SEC-EDGAR corporate filings using the nanoGPT framework. Trained on a single NVIDIA RTX 4070 GPU in approximately 8 hours, reaching a final validation loss of 2.28.</p>
<h2>Key Numbers</h2>
<table>
  <tr><td class="key">Parameters</td><td>124M</td></tr>
  <tr><td class="key">Training tokens</td><td>1.55B</td></tr>
  <tr><td class="key">Validation loss</td><td>2.28</td></tr>
  <tr><td class="key">Training steps</td><td>47,000</td></tr>
  <tr><td class="key">Training time</td><td>~8 hours</td></tr>
  <tr><td class="key">Hardware</td><td>NVIDIA RTX 4070 (12GB)</td></tr>
  <tr><td class="key">Framework</td><td>nanoGPT</td></tr>
</table>
<h2>Findings</h2>
<ul>
  <li>Model learns SEC document structure, financial vocabulary, and boilerplate language</li>
  <li>Excels at echoing input content (tables, numbers, formatting)</li>
  <li>Struggles with numerical consistency and long-range coherence</li>
  <li>Exhibits loop attractors on high-probability SEC phrases</li>
  <li>Can draft placeholder text resembling authentic filings</li>
</ul>
<h2>Citation</h2>
<pre style="background:#f5f5f5; padding:1rem; border-radius:6px; overflow-x:auto; font-size:0.9rem;">
@article{li2026secedgargpt,
  title={SEC-EDGAR-GPT: A 124M Financial Language Model},
  author={Li, Zhiwei},
  year={2026}
}
</pre>
<div class="footer">
  <p>Zhiwei Li &middot; <a href="mailto:lzwjava@gmail.com">lzwjava@gmail.com</a> &middot; June 2026</p>
</div>
</div>
</body>
</html>`;

export default {
  async fetch() {
    return new Response(HTML, {
      headers: { "Content-Type": "text/html;charset=utf-8" },
    });
  },
};
