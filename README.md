# ProWriteBench

**A frontier evaluation benchmark for LLM professional writing capabilities**

ProWriteBench evaluates large language models on realistic professional writing scenarios, focusing on business communication, documentation, and stakeholder management. Unlike existing benchmarks that focus on creative writing, ProWriteBench tests models on the challenging aspects of professional communication: balancing competing stakeholder needs, satisfying hard constraints, and navigating iterative feedback.

## Why ProWriteBench?

**Fills a Critical Gap**: Most writing benchmarks (EQ-Bench, WritingBench) focus on creative writing. Professional business writing evaluation is underserved.

**Exposes Weaknesses**: Following EQ-Bench's philosophy, ProWriteBench deliberately targets common LLM failure modes rather than showcasing strengths.

**Novel Evaluation Metrics**:
- **Constraint Satisfaction**: Binary checks for hard requirements (word limits, required elements, forbidden content)
- **Audience-Specific Clarity**: Tests comprehension with simulated reader profiles
- **Multi-Stakeholder Balance**: Measures if all parties' needs are addressed fairly
- **Revision Coherence**: Tracks quality across feedback iterations
- **Professional Appropriateness**: Judge evaluation for tone and diplomatic language

## Task Categories

### 1. Multi-Stakeholder Communication (10 tasks)
Write for audiences with conflicting interests.

**Example**: Layoff announcement email balancing transparency for affected employees, reassurance for remaining staff, and leadership optics for executives.

**Why it's hard**: LLMs often optimize for one stakeholder or produce bland, non-committal text.

### 2. Constrained Revision (10 tasks)
Navigate contradictory feedback without overcorrecting.

**Example**: Email revised as "too formal" → "now too casual" → "maintain professionalism but be warmer"

**Why it's hard**: Tests self-correction and ability to balance competing guidance.

### 3. Implicit Requirements (10 tasks)
Extract unstated professional norms from context.

**Example**: "Decline this partnership" → must infer: be diplomatic, don't burn bridges, cite valid reasons, leave door open.

**Why it's hard**: Professional writing requires reading between the lines. LLMs often take requests literally.

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/prowritebench.git
cd prowritebench
pip install -e .
```

### Set Up API Keys

```bash
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

### Evaluate a Single Task

```bash
python examples/evaluate_single_task.py --task MS-001 --model claude-opus-4-5
```

### Run Full Benchmark

```bash
python examples/run_benchmark.py --model claude-opus-4-5 --output results/
```

### Generate Report

```bash
python examples/generate_report.py --results results/claude-opus-4-5.json
```

## Scoring System

Each task is scored 0-100 across five dimensions:

1. **Constraint Satisfaction (30%)**: Binary checks for hard requirements
2. **Audience Clarity (25%)**: Comprehension tests with reader simulations
3. **Stakeholder Balance (20%)**: Multi-perspective satisfaction
4. **Professional Appropriateness (15%)**: Judge evaluation
5. **Revision Coherence (10%)**: Quality improvement across iterations

**Critical Failure Penalty**: Tasks with inappropriate tone, missing mandatory elements, or factual errors receive a 0.5x multiplier per failure.

## Example Task: MS-001 Layoff Announcement

**Scenario**: You're VP of Engineering at a tech startup. The board decided to reduce headcount by 15% due to funding challenges. Write an all-hands email.

**Stakeholders**:
- Affected employees: need transparency, respect, support
- Remaining employees: need reassurance, clarity on future
- Executive team: need to maintain confidence, show leadership

**Constraints**:
- 250-400 words
- Must include: reason for decision, timeline, support offered, future outlook
- Cannot include: names of affected individuals, specific dollar figures
- Tone: Empathetic but professional, transparent but forward-looking

**Critical Failures**: Overly casual tone, missing support resources, blame-shifting language

## Custom Model Integration

ProWriteBench supports any LLM through the abstract model interface:

```python
from src.models.base import BaseModel

class CustomModel(BaseModel):
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        # Your implementation
        pass
```

See [src/models/base.py](src/models/base.py) for details.

## Project Structure

```
prowritebench/
├── data/
│   ├── tasks/              # 30 task definitions (JSON)
│   └── rubrics/            # Evaluation rubrics and judge prompts
├── src/
│   ├── benchmark.py        # Main orchestration
│   ├── tasks.py            # Task loading
│   ├── evaluators/         # 5 evaluation modules
│   ├── models/             # Model adapters (OpenAI, Anthropic, custom)
│   └── utils/              # Scoring and bias mitigation
├── examples/               # CLI tools for running benchmarks
└── results/                # Output directory
```

## Design Principles

1. **Realistic Constraints**: Professional writing has hard requirements (word limits, tone, required elements)
2. **Competing Priorities**: Real scenarios involve stakeholders with conflicting needs
3. **Iterative Nature**: Professional writing is rarely one-shot; revision tasks test adaptation
4. **Implicit Norms**: Professional communication has unstated rules that must be inferred
5. **Critical Failures**: Some mistakes (inappropriate tone, missing key info) should severely impact scores

## Bias Mitigation

Following EQ-Bench Creative Writing v3 principles:
- Truncate outputs at 500 words for fair comparison
- Randomize position for pairwise judgments
- Use multiple judge models and ensemble
- Include human calibration samples

## Citation

If you use ProWriteBench in your research, please cite:

```bibtex
@misc{prowritebench2025,
  title={ProWriteBench: A Frontier Evaluation Benchmark for LLM Professional Writing},
  author={Samrath Singh Patpatia},
  year={2026},
  url={https://github.com/samrathp29/ProWriteBench}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Inspired by:
- [EQ-Bench Creative Writing v3](https://eqbench.com/creative_writing_longform.html)
- [FrontierMath](https://epoch.ai/frontiermath)
- Research on LLM evaluation methodologies
