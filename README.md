# 🏥 Intelligent OR Scheduling & Optimization System

## 🌟 Project Overview: Solving the Hospital Bottleneck

**Problem:** Manual scheduling of Operating Rooms (ORs) and surgical teams is a critical bottleneck in healthcare, leading to lost revenue, resource waste, and unnecessary patient delays.

**Our Solution:** This project delivers an algorithmic engine built in **Python** that automatically generates highly efficient, conflict-free surgical schedules. We achieve true optimization by embedding a **priority-driven triage system** and **specific resource constraints** directly into the scheduling logic.

**Result:** A quantifiable reduction in resource waste and a smoother, more reliable patient flow, demonstrated through tangible performance metrics.

## 🎯 Key Features & Algorithmic Power

| Feature | Description | Algorithmic Core |
| :--- | :--- | :--- |
| **Surgical Triage** | Surgeries are automatically sorted based on clinical urgency (Emergency > Urgent > Elective). | **Min-Heap (Priority Queue)** |
| **Custom Seniority Tie-Breaker** | Within the same clinical priority (e.g., two Urgent cases), the **most senior surgeon's case is scheduled first** to optimize senior resource usage. | **Custom `__lt__` comparison logic** |
| **Dynamic Conflict Detection** | Prevents double-booking of surgeons and ensures all schedule windows respect unique, custom surgeon shifts (e.g., Dr. Lee's 14:00 cutoff). | **Hash Maps** for O(1) availability lookup |
| **Maximized Efficiency** | Greedy strategy attempts to fill the earliest feasible slot, maximizing case density. | **Greedy Algorithm** |
| **Measurable Impact** | Outputs clear metrics on **OR Utilization Rate (CDI)** and **Total Idle Time**. | **Prefix Sum / Data Analysis** |

## ⚙️ How to Run the Project

### Prerequisites

* **Python 3.x** installed

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/soniyakuchekar71/CV-PROJECT](https://github.com/soniyakuchekar71/CV-PROJECT)
    cd CV-PROJECT
    ```

### Usage

1.  The core logic is contained in `main.py`. You can modify the resource lists and surgical cases within the `if __name__ == "__main__":` block to test different scenarios.
2.  Run the simulation from your terminal:
    ```bash
    python main.py
    ```

### Sample Output Interpretation

The output displays the final schedule, followed by the metrics. Pay close attention to the unscheduled cases, as these directly highlight capacity and resource constraint limits (e.g., when Dr. Lee's custom shift time blocks a later assignment).

***

## 🧠 Design & Custom Constraints (Crucial for Evaluation)

The success of this optimization lies in its ability to handle non-uniform, real-world constraints:

1.  **Unique Operating Hours:** The OR day runs from **7:30 AM to 4:30 PM**, deviating from a standard 8-hour window.
2.  **Surgeon-Specific Shifts:** The scheduler respects custom availability (e.g., **Dr. Lee** is unavailable after **14:00** due to a mandatory training session).
3.  **Advanced Priority Triage (`__lt__` function):**
    * **Level 1:** `Emergency`
    * **Level 2 (Tie-breaker):** `Seniority` (Lower number = more senior)
    * **Level 3 (Final Tie-breaker):** `Shortest Duration`

