import heapq
import datetime
import sys

# --- Operational Setup: Defining the Hospital's World ---

# Standard shift times for the Operating Rooms (ORs)
OR_START_HOUR = 7  
OR_START_MINUTE = 30 # We start early to maximize capacity!
OR_END_HOUR = 16 
OR_END_MINUTE = 30 # Day ends at 4:30 PM

# Surgeon-specific working windows: This is a critical custom constraint
# and deviation from generic scheduling. It forces the algorithm to check 
# individual limits, not just the OR's open hours.
CUSTOM_SURGEON_HOURS = {
    "Dr. Smith": (7, 30, 16, 30), # Full, standard shift
    "Dr. Jones": (8, 0, 16, 0),   # Slightly later start/earlier finish
    "Dr. Lee": (7, 30, 14, 0),    # Custom rule: Must finish by 2:00 PM for training/meeting!
    "Dr. Patel": (7, 30, 16, 30),
    "Dr. Singh": (8, 0, 15, 0),   # Short shift
    "Dr. Gupta": (7, 30, 14, 30),
}


class Surgery:
    """
    A single surgical case object. This object holds all the data needed for 
    triage and scheduling decisions. The magic here is the __lt__ method, 
    which defines the priority queue's internal sorting logic.
    """
    def __init__(self, s_id, duration_minutes, surgeon_id, priority, surgeon_seniority):
        self.s_id = s_id
        self.duration = duration_minutes
        self.surgeon_id = surgeon_id
        self.priority = priority
        # Custom data point: Critical for the tie-breaker logic
        self.surgeon_seniority = surgeon_seniority 
        self.scheduled_or = None
        self.scheduled_start_time = None
        self.scheduled_end_time = None

    def get_priority_value(self):
        # Maps clinical labels to a simple, sortable numerical triage score.
        if self.priority == "Emergency": return 1
        if self.priority == "Urgent": return 2
        return 3 # Elective cases are last in line

    # --- CORE LOGIC: DEFINES THE TRIAGE SYSTEM FOR THE MIN-HEAP ---
    def __lt__(self, other):
        # 1. Triage Rule (Highest Priority): Emergency always comes before Urgent.
        if self.get_priority_value() != other.get_priority_value():
            return self.get_priority_value() < other.get_priority_value()
        
        # 2. Custom Tie-breaker (Seniority Rule): Within the same priority group, 
        #    we schedule the most senior surgeon's case first to respect their time.
        if self.surgeon_seniority != other.surgeon_seniority:
            return self.surgeon_seniority < other.surgeon_seniority

        # 3. Final Tie-breaker (Efficiency Rule): If priorities and seniority are equal, 
        #    we prioritize shorter cases. This is a common greedy strategy to prevent 
        #    a very long case from locking up the start of the day unnecessarily.
        return self.duration < other.duration 

    def __repr__(self):
        start = self.scheduled_start_time.strftime("%H:%M") if self.scheduled_start_time else "N/A"
        end = self.scheduled_end_time.strftime("%H:%M") if self.scheduled_end_time else "N/A"
        return f"Case {self.s_id} | {self.surgeon_id} | {self.priority} | {start}-{end} | {self.duration}m"


class Scheduler:
    """
    The main engine. It manages resource tracking and implements the Greedy 
    Scheduling Algorithm using Hash Maps for O(1) resource lookups.
    """
    def __init__(self, or_ids, surgeon_ids):
        # Initializing resource availability to the start of the day.
        # This acts as a Hash Map (dictionary) for O(1) resource tracking.
        self.ors = {or_id: self._day_start() for or_id in or_ids}
        self.surgeons = {surgeon_id: self._day_start() for surgeon_id in surgeon_ids}
        self.pending_surgeries = [] # This is the Min-Heap
        self.scheduled_surgeries = []
        self.unscheduled_surgeries = []

    # Helper methods for consistent time reference
    def _day_start(self, hour=OR_START_HOUR, minute=OR_START_MINUTE):
        today = datetime.date.today()
        return datetime.datetime(today.year, today.month, today.day, hour, minute)

    def _day_end(self, hour=OR_END_HOUR, minute=OR_END_MINUTE):
        today = datetime.date.today()
        return datetime.datetime(today.year, today.month, today.day, hour, minute)

    def add_surgery(self, surgery):
        # Standard procedure: Dump the new case into the priority queue.
        heapq.heappush(self.pending_surgeries, surgery)

    def _surgeon_shift_limits(self, surgeon_id):
        # Fetches the specific, potentially non-standard, shift limits for the surgeon.
        h_start, m_start, h_end, m_end = CUSTOM_SURGEON_HOURS.get(
            surgeon_id, (OR_START_HOUR, OR_START_MINUTE, OR_END_HOUR, OR_END_MINUTE)
        )
        return self._day_start(h_start, m_start), self._day_end(h_end, m_end)

    def _find_earliest_feasible_slot(self, surgery):
        """
        This is the inner loop of the greedy algorithm. We search for the 
        absolute earliest time that respects ALL constraints (OR, Surgeon, and Time limits).
        """
        surgeon_id = surgery.surgeon_id
        shift_start, shift_end = self._surgeon_shift_limits(surgeon_id)

        # Initialize the 'best time' far into the future.
        earliest_start = self._day_end() + datetime.timedelta(days=1)
        best_or = None

        for or_id, or_available in self.ors.items():
            surgeon_available = self.surgeons.get(surgeon_id, shift_start)

            # CRITICAL STEP: The start time is determined by the latest availability 
            # among the OR, the Surgeon, AND the Surgeon's custom shift start.
            possible_start = max(or_available, surgeon_available, shift_start)
            possible_end = possible_start + datetime.timedelta(minutes=surgery.duration)

            # CONSTRAINT CHECK: Must finish before OR closes AND before surgeon leaves (e.g., Dr. Lee's 14:00 constraint).
            if possible_end <= self._day_end() and possible_end <= shift_end:
                
                # We found a valid slot. Is it better (earlier) than our current best?
                if possible_start < earliest_start:
                    earliest_start = possible_start
                    best_or = or_id

        if best_or:
            return best_or, earliest_start
        return None, None

    def triage_and_optimize_or_flow(self):
        """
        The main control loop. This runs in O(N log N * O) time, where N is cases, 
        and O is OR count. The log N comes from the Heap operation.
        """
        while self.pending_surgeries:
            # 1. TRIAGE: Get the highest priority case, determined by the __lt__ rules.
            case = heapq.heappop(self.pending_surgeries)
            
            # 2. GREEDY ASSIGNMENT: Find the slot.
            or_id, start = self._find_earliest_feasible_slot(case)

            if or_id and start:
                # Success! Record the assignment.
                case.scheduled_or = or_id
                case.scheduled_start_time = start
                case.scheduled_end_time = start + datetime.timedelta(minutes=case.duration)

                # Update the Hash Maps to reflect the new resource busy-time.
                self.ors[or_id] = case.scheduled_end_time
                self.surgeons[case.surgeon_id] = case.scheduled_end_time

                self.scheduled_surgeries.append(case)
            else:
                # Failure! The case could not be placed due to capacity or constraint limits.
                self.unscheduled_surgeries.append(case)

        # Order the final output chronologically for human readability.
        self.scheduled_surgeries.sort(key=lambda s: s.scheduled_start_time)

    # --- Output and Metrics ---
    def _format_duration(self, minutes):
        # Utility function to make output friendlier (e.g., 90m -> 1h 30m)
        if minutes < 60:
            return f"{minutes}m"
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h{(' ' + str(mins) + 'm') if mins else ''}"

    def print_schedule(self):
        today_str = datetime.date.today().strftime("%A, %B %d, %Y")
        print("=" * 80)
        print(f"Schedule for {today_str}")
        print("=" * 80)
        print("\nHere's the day's plan (ordered by scheduled start time):\n")

        if not self.scheduled_surgeries:
            print("No cases could be scheduled today.")
        else:
            for case in self.scheduled_surgeries:
                start = case.scheduled_start_time.strftime("%H:%M")
                end = case.scheduled_end_time.strftime("%H:%M")
                dur = self._format_duration(case.duration)
                print(f"- OR {case.scheduled_or} | {start} - {end} | Case {case.s_id} | {case.surgeon_id} | "
                      f"{case.priority} | {dur}")

        if self.unscheduled_surgeries:
            print("\nUnscheduled cases (couldn't fit within OR or surgeon hours):")
            for case in self.unscheduled_surgeries:
                print(f"- Case {case.s_id} | {case.surgeon_id} | {case.priority} | {self._format_duration(case.duration)}")

        # Summary metrics in plain language
        print("\nSummary & Utilization (CDI - Case Density Index):")
        total_minutes_available = (self._day_end() - self._day_start()).total_seconds() / 60
        for or_id in self.ors:
            scheduled_minutes = sum(s.duration for s in self.scheduled_surgeries if s.scheduled_or == or_id)
            idle = total_minutes_available - scheduled_minutes
            utilization = (scheduled_minutes / total_minutes_available) * 100 if total_minutes_available else 0
            print(f"  {or_id}: {scheduled_minutes} minutes scheduled, {idle} minutes idle ({utilization:.1f}% used)")

        print("\nTotals:")
        print(f"  Scheduled cases: {len(self.scheduled_surgeries)}")
        print(f"  Unscheduled cases: {len(self.unscheduled_surgeries)}")
        print("=" * 80)


if __name__ == "__main__":
    # Robust check for encoding needed only on specific Windows environments
    if sys.platform == "win32" and sys.version_info >= (3, 7):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # --- DEFINE HOSPITAL RESOURCES ---
    or_rooms = ["OR-A", "OR-B"]
    surgeons_available = ["Dr. Smith", "Dr. Jones", "Dr. Lee", "Dr. Patel", "Dr. Singh", "Dr. Gupta"]

    scheduler = Scheduler(or_rooms, surgeons_available)

    # --- DEFINE SURGICAL CASES ---
    # Surgery(ID, Duration, Surgeon, Priority, Seniority)
    
    # Priority 1: Emergency (High priority, Seniority breaks ties)
    scheduler.add_surgery(Surgery(108, 60, "Dr. Smith", "Emergency", 1)) 
    scheduler.add_surgery(Surgery(105, 30, "Dr. Jones", "Emergency", 2))
    scheduler.add_surgery(Surgery(112, 30, "Dr. Gupta", "Emergency", 3))

    # Priority 2: Urgent
    scheduler.add_surgery(Surgery(102, 90, "Dr. Jones", "Urgent", 2))
    scheduler.add_surgery(Surgery(110, 45, "Dr. Patel", "Urgent", 2)) # Same seniority as Jones; shorter duration will place it before 102
    scheduler.add_surgery(Surgery(106, 75, "Dr. Lee", "Urgent", 3))

    # Priority 3: Elective (Lowest Priority)
    scheduler.add_surgery(Surgery(101, 60, "Dr. Smith", "Elective", 1)) 
    scheduler.add_surgery(Surgery(104, 120, "Dr. Smith", "Elective", 1)) # Very long case, challenging to fit
    scheduler.add_surgery(Surgery(107, 180, "Dr. Jones", "Elective", 2))
    scheduler.add_surgery(Surgery(113, 60, "Dr. Patel", "Elective", 2))
    scheduler.add_surgery(Surgery(103, 45, "Dr. Lee", "Elective", 3))   # Dr. Lee case, restricted to 14:00
    scheduler.add_surgery(Surgery(109, 60, "Dr. Lee", "Elective", 3))   # Dr. Lee case, very likely to be unscheduled
    scheduler.add_surgery(Surgery(111, 90, "Dr. Singh", "Elective", 4)) # Lowest seniority, scheduled last

    # --- EXECUTE THE SCHEDULING ALGORITHM ---
    scheduler.triage_and_optimize_or_flow()
    scheduler.print_schedule()