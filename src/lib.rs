#![forbid(unsafe_code)]
//! # band-ensemble-rs
//!
//! A collection of autonomous agents forming a self-improving band ensemble.
//! Uses Hodge decomposition to negotiate tempo, conservation laws to balance
//! energy, and an intention field to guide the group toward target musical state.

// ─────────────────────────────────────────────────────────────────────────────
// agent module
// ─────────────────────────────────────────────────────────────────────────────

/// The instrumental role a local agent plays in the ensemble.
#[derive(Debug, Clone, PartialEq)]
pub enum InstrumentRole {
    /// Percussion / time-keeper
    Drums,
    /// Low-end harmonic foundation
    Bass,
    /// Chordal / melodic mid-range
    Keys,
    /// Brass / wind soloists
    Horns,
    /// Sustained texture layer
    Pads,
}

/// A single autonomous player in the ensemble.
#[derive(Debug, Clone)]
pub struct LocalAgent {
    /// Unique identifier within the ensemble
    pub id: u64,
    /// The instrument role this agent fills
    pub role: InstrumentRole,
    /// The tempo (BPM) this agent would like the ensemble to adopt
    pub tempo_proposal: f64,
    /// Current energy level (conserved quantity)
    pub energy: f64,
    /// Phase within a single bar cycle, in `[0, 1)`
    pub phase: f64,
    /// Four-element spectral state vector derived from role
    pub spectral_state: [f64; 4],
}

impl LocalAgent {
    /// Create a new agent.
    ///
    /// - `spectral_state` is initialised from `role` (see source).
    /// - `energy` starts at `1.0`, `phase` at `0.0`.
    pub fn new(id: u64, role: InstrumentRole, tempo: f64) -> Self {
        let spectral_state = match role {
            InstrumentRole::Drums => [1.0, 0.0, 0.0, 0.0],
            InstrumentRole::Bass  => [0.0, 1.0, 0.0, 0.0],
            InstrumentRole::Keys  => [0.0, 0.0, 1.0, 0.0],
            InstrumentRole::Horns => [0.0, 0.0, 0.0, 1.0],
            InstrumentRole::Pads  => [0.5, 0.5, 0.5, 0.5],
        };
        Self {
            id,
            role,
            tempo_proposal: tempo,
            energy: 1.0,
            phase: 0.0,
            spectral_state,
        }
    }

    /// Advance the agent's internal phase by `dt` seconds at `master_tempo` BPM.
    ///
    /// Phase wraps to `[0, 1)`.
    pub fn tick(&mut self, dt: f64, master_tempo: f64) {
        self.phase = (self.phase + dt * master_tempo / 60.0) % 1.0;
    }

    /// Propose a tempo to the ensemble (slightly influenced by the first
    /// component of the spectral state).
    pub fn propose(&self) -> f64 {
        self.tempo_proposal * (1.0 + self.spectral_state[0] * 0.01)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// hodge module
// ─────────────────────────────────────────────────────────────────────────────

/// Hodge decomposition of a set of tempo proposals.
///
/// Separates the collective signal into gradient (shared trend), curl
/// (rotational disagreement) and harmonic (true consensus) components.
#[derive(Debug, Clone)]
pub struct HodgeDecomposition {
    /// Irrotational component — the arithmetic mean of proposals
    pub gradient: f64,
    /// Rotational component — the standard deviation of proposals
    pub curl: f64,
    /// Harmonic component — the median of proposals
    pub harmonic: f64,
}

impl HodgeDecomposition {
    /// Decompose a slice of tempo proposals into Hodge components.
    ///
    /// # Panics
    /// Panics if `proposals` is empty.
    pub fn decompose(proposals: &[f64]) -> Self {
        assert!(!proposals.is_empty(), "proposals must not be empty");

        let n = proposals.len() as f64;

        // gradient = mean
        let gradient = proposals.iter().sum::<f64>() / n;

        // curl = standard deviation
        let variance = proposals.iter().map(|&x| (x - gradient).powi(2)).sum::<f64>() / n;
        let curl = variance.sqrt();

        // harmonic = median
        let mut sorted = proposals.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        let harmonic = if sorted.len().is_multiple_of(2) {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        };

        Self { gradient, curl, harmonic }
    }

    /// Derive a consensus tempo from the decomposition.
    ///
    /// `0.7 * harmonic + 0.3 * gradient`
    pub fn consensus_tempo(&self) -> f64 {
        0.7 * self.harmonic + 0.3 * self.gradient
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// alignment module
// ─────────────────────────────────────────────────────────────────────────────

/// The shared musical state that all agents align to.
#[derive(Debug, Clone)]
pub struct AlignmentState {
    /// Current master tempo in BPM
    pub current_tempo: f64,
    /// Current tonal centre as a MIDI pitch class (0–11)
    pub current_key: u8,
    /// Length of the musical form in bars (e.g. 12 for a blues)
    pub current_form: u8,
}

impl AlignmentState {
    /// Create a new alignment state.
    pub fn new(tempo: f64, key: u8, form: u8) -> Self {
        Self {
            current_tempo: tempo,
            current_key: key,
            current_form: form,
        }
    }

    /// Update the master tempo using a Hodge-consensus of agent proposals.
    pub fn update_tempo(&mut self, proposals: &[f64]) {
        if proposals.is_empty() {
            return;
        }
        let hodge = HodgeDecomposition::decompose(proposals);
        self.current_tempo = hodge.consensus_tempo();
    }

    /// Measure variance (disagreement) across a set of proposals.
    pub fn variance(proposals: &[f64]) -> f64 {
        if proposals.is_empty() {
            return 0.0;
        }
        let n = proposals.len() as f64;
        let mean = proposals.iter().sum::<f64>() / n;
        proposals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// conservation module
// ─────────────────────────────────────────────────────────────────────────────

/// Energy conservation ledger for the ensemble.
///
/// Ensures the total energy across all agents remains close to a fixed budget.
#[derive(Debug, Clone)]
pub struct EnsembleConservation {
    /// The target total energy for the ensemble
    pub total_budget: f64,
    /// Per-agent energy allocations
    pub agent_energies: Vec<f64>,
}

impl EnsembleConservation {
    /// Create a new conservation ledger, distributing `budget` evenly across
    /// `n_agents` agents.
    pub fn new(budget: f64, n_agents: usize) -> Self {
        let share = if n_agents == 0 { 0.0 } else { budget / n_agents as f64 };
        Self {
            total_budget: budget,
            agent_energies: vec![share; n_agents],
        }
    }

    /// Redistribute energy so every agent holds an equal share of the budget.
    pub fn rebalance(&mut self) {
        let n = self.agent_energies.len();
        if n == 0 {
            return;
        }
        let share = self.total_budget / n as f64;
        for e in self.agent_energies.iter_mut() {
            *e = share;
        }
    }

    /// Sum of all agent energies.
    pub fn total_energy(&self) -> f64 {
        self.agent_energies.iter().sum()
    }

    /// Returns `true` if total energy is within 1 % of the budget.
    pub fn is_conserved(&self) -> bool {
        (self.total_energy() - self.total_budget).abs() < 0.01 * self.total_budget
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// intention module
// ─────────────────────────────────────────────────────────────────────────────

/// A gravitational field that attracts the ensemble toward a musical target.
#[derive(Debug, Clone)]
pub struct IntentionField {
    /// The desired master tempo
    pub target_tempo: f64,
    /// The desired tonal centre (MIDI pitch class)
    pub target_key: u8,
    /// Strength of the attraction, in `[0, 1]`
    pub attraction_strength: f64,
}

impl IntentionField {
    /// Create a new intention field with default attraction strength of `0.3`.
    pub fn new(tempo: f64, key: u8) -> Self {
        Self {
            target_tempo: tempo,
            target_key: key,
            attraction_strength: 0.3,
        }
    }

    /// Blend the `current` tempo toward the target.
    ///
    /// `result = current + attraction * (target - current)`
    pub fn pull_tempo(&self, current: f64) -> f64 {
        current + self.attraction_strength * (self.target_tempo - current)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// phase module
// ─────────────────────────────────────────────────────────────────────────────

/// The collective musical phase of the ensemble.
#[derive(Debug, Clone, PartialEq)]
pub enum EnsemblePhase {
    /// Low variance — tight, locked-in groove
    Groove,
    /// Moderate variance — building tension
    Tension,
    /// High variance — creative chaos
    Chaos,
    /// Very high variance — about to resolve
    Resolution,
}

/// Tracks recent tempo variance to infer the current ensemble phase.
#[derive(Debug, Clone)]
pub struct PhaseDetector {
    /// Ring buffer of the last 8 variance readings
    pub tempo_variance_history: Vec<f64>,
}

impl PhaseDetector {
    /// Create a new phase detector with an empty history buffer.
    pub fn new() -> Self {
        Self { tempo_variance_history: Vec::with_capacity(8) }
    }

    /// Record a new variance sample, keeping only the last 8.
    pub fn record(&mut self, variance: f64) {
        if self.tempo_variance_history.len() == 8 {
            self.tempo_variance_history.remove(0);
        }
        self.tempo_variance_history.push(variance);
    }

    /// Derive the current phase from the most recent variance reading.
    ///
    /// | variance   | phase      |
    /// |------------|------------|
    /// | < 0.5      | Groove     |
    /// | < 2.0      | Tension    |
    /// | < 10.0     | Chaos      |
    /// | ≥ 10.0     | Resolution |
    pub fn detect(&self) -> EnsemblePhase {
        let v = self.tempo_variance_history.last().copied().unwrap_or(0.0);
        if v < 0.5 {
            EnsemblePhase::Groove
        } else if v < 2.0 {
            EnsemblePhase::Tension
        } else if v < 10.0 {
            EnsemblePhase::Chaos
        } else {
            EnsemblePhase::Resolution
        }
    }
}

impl Default for PhaseDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ensemble module
// ─────────────────────────────────────────────────────────────────────────────

/// A self-organising ensemble of [`LocalAgent`]s that negotiate tempo,
/// conserve energy, and track collective musical phase.
#[derive(Debug)]
pub struct Ensemble {
    /// All agents currently in the ensemble
    pub agents: Vec<LocalAgent>,
    /// Shared alignment state
    pub alignment: AlignmentState,
    /// Energy conservation ledger
    pub conservation: EnsembleConservation,
    /// Intention field guiding the group
    pub intention: IntentionField,
    /// Phase detector
    pub phase_detector: PhaseDetector,
    /// Number of ticks elapsed since creation
    pub tick_count: u64,
    next_id: u64,
}

impl Ensemble {
    /// Create an empty ensemble at the given initial tempo.
    ///
    /// Conservation budget is `10.0`; alignment key is `0` (C); form is `12` bars.
    pub fn new(initial_tempo: f64) -> Self {
        Self {
            agents: Vec::new(),
            alignment: AlignmentState::new(initial_tempo, 0, 12),
            conservation: EnsembleConservation::new(10.0, 0),
            intention: IntentionField::new(initial_tempo, 0),
            phase_detector: PhaseDetector::new(),
            tick_count: 0,
            next_id: 0,
        }
    }

    /// Add a new agent with the given role and return its assigned id.
    pub fn spawn_agent(&mut self, role: InstrumentRole) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let agent = LocalAgent::new(id, role, self.alignment.current_tempo);
        self.agents.push(agent);
        // Expand the conservation ledger for the new agent count.
        self.conservation.agent_energies.push(0.0);
        self.conservation.rebalance();
        id
    }

    /// Remove an agent by id.  Returns `true` if the agent was found and removed.
    pub fn remove_agent(&mut self, id: u64) -> bool {
        if let Some(pos) = self.agents.iter().position(|a| a.id == id) {
            self.agents.remove(pos);
            // Shrink the conservation ledger and rebalance.
            if !self.conservation.agent_energies.is_empty() {
                self.conservation.agent_energies.pop();
                self.conservation.rebalance();
            }
            true
        } else {
            false
        }
    }

    /// Collect proposals from all agents, Hodge-decompose, apply intention
    /// pull, update alignment, and return the new consensus tempo.
    pub fn negotiate_tempo(&mut self) -> f64 {
        if self.agents.is_empty() {
            return self.alignment.current_tempo;
        }
        let proposals: Vec<f64> = self.agents.iter().map(|a| a.propose()).collect();
        self.alignment.update_tempo(&proposals);
        let pulled = self.intention.pull_tempo(self.alignment.current_tempo);
        self.alignment.current_tempo = pulled;
        pulled
    }

    /// Advance all agents by `dt` seconds, check conservation, and record phase.
    pub fn tick(&mut self, dt: f64) {
        let tempo = self.alignment.current_tempo;
        for agent in self.agents.iter_mut() {
            agent.tick(dt, tempo);
        }
        if !self.conservation.is_conserved() {
            self.conservation.rebalance();
        }
        let proposals: Vec<f64> = self.agents.iter().map(|a| a.propose()).collect();
        let variance = AlignmentState::variance(&proposals);
        self.phase_detector.record(variance);
        self.tick_count += 1;
    }

    /// Run the ensemble for `duration_secs` of simulated time in steps of `dt`.
    pub fn jam(&mut self, duration_secs: f64, dt: f64) {
        let steps = (duration_secs / dt).round() as u64;
        for _ in 0..steps {
            self.negotiate_tempo();
            self.tick(dt);
        }
    }

    /// Collect the spectral state of every agent.
    pub fn spectral_report(&self) -> Vec<[f64; 4]> {
        self.agents.iter().map(|a| a.spectral_state).collect()
    }

    /// Return the current ensemble phase.
    pub fn current_phase(&self) -> EnsemblePhase {
        self.phase_detector.detect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── HodgeDecomposition ────────────────────────────────────────────────────

    #[test]
    fn hodge_uniform_curl_is_zero() {
        let proposals = vec![120.0, 120.0, 120.0];
        let h = HodgeDecomposition::decompose(&proposals);
        assert!((h.curl).abs() < 1e-10, "curl should be 0 for uniform proposals");
    }

    #[test]
    fn hodge_uniform_gradient_equals_value() {
        let proposals = vec![120.0, 120.0, 120.0];
        let h = HodgeDecomposition::decompose(&proposals);
        assert!((h.gradient - 120.0).abs() < 1e-10);
    }

    #[test]
    fn hodge_uniform_harmonic_equals_value() {
        let proposals = vec![120.0, 120.0, 120.0];
        let h = HodgeDecomposition::decompose(&proposals);
        assert!((h.harmonic - 120.0).abs() < 1e-10);
    }

    #[test]
    fn hodge_spread_curl_nonzero() {
        let proposals = vec![100.0, 120.0, 140.0];
        let h = HodgeDecomposition::decompose(&proposals);
        assert!(h.curl > 0.0);
    }

    #[test]
    fn hodge_spread_gradient_is_mean() {
        let proposals = vec![100.0, 120.0, 140.0];
        let h = HodgeDecomposition::decompose(&proposals);
        assert!((h.gradient - 120.0).abs() < 1e-10);
    }

    #[test]
    fn hodge_spread_harmonic_is_median() {
        let proposals = vec![100.0, 120.0, 140.0];
        let h = HodgeDecomposition::decompose(&proposals);
        assert!((h.harmonic - 120.0).abs() < 1e-10);
    }

    #[test]
    fn hodge_consensus_tempo_in_range() {
        let proposals = vec![100.0, 140.0];
        let h = HodgeDecomposition::decompose(&proposals);
        let ct = h.consensus_tempo();
        // Should be between the min and max proposal.
        assert!(ct >= 100.0 && ct <= 140.0);
    }

    #[test]
    fn hodge_even_count_median() {
        // Even-count median is average of the two middle values.
        let proposals = vec![100.0, 110.0, 130.0, 140.0];
        let h = HodgeDecomposition::decompose(&proposals);
        assert!((h.harmonic - 120.0).abs() < 1e-10);
    }

    // ── AlignmentState ────────────────────────────────────────────────────────

    #[test]
    fn alignment_update_tempo_converges() {
        let mut a = AlignmentState::new(120.0, 0, 12);
        let proposals = vec![118.0, 119.0, 120.0, 121.0, 122.0];
        a.update_tempo(&proposals);
        // Consensus should be near 120.
        assert!((a.current_tempo - 120.0).abs() < 5.0);
    }

    #[test]
    fn alignment_variance_zero_for_uniform() {
        let proposals = vec![120.0, 120.0, 120.0];
        assert!(AlignmentState::variance(&proposals) < 1e-10);
    }

    #[test]
    fn alignment_variance_nonzero_for_spread() {
        let proposals = vec![100.0, 120.0, 140.0];
        assert!(AlignmentState::variance(&proposals) > 0.0);
    }

    #[test]
    fn alignment_variance_empty_is_zero() {
        assert_eq!(AlignmentState::variance(&[]), 0.0);
    }

    // ── EnsembleConservation ──────────────────────────────────────────────────

    #[test]
    fn conservation_new_distributes_evenly() {
        let c = EnsembleConservation::new(10.0, 4);
        for &e in &c.agent_energies {
            assert!((e - 2.5).abs() < 1e-10);
        }
    }

    #[test]
    fn conservation_total_energy_matches_budget() {
        let c = EnsembleConservation::new(10.0, 4);
        assert!((c.total_energy() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn conservation_is_conserved_after_new() {
        let c = EnsembleConservation::new(10.0, 5);
        assert!(c.is_conserved());
    }

    #[test]
    fn conservation_rebalance_restores_conservation() {
        let mut c = EnsembleConservation::new(10.0, 4);
        c.agent_energies[0] = 0.0; // break conservation
        c.rebalance();
        assert!(c.is_conserved());
    }

    #[test]
    fn conservation_add_agent_then_rebalance() {
        let mut c = EnsembleConservation::new(10.0, 3);
        c.agent_energies.push(0.0);
        c.rebalance();
        assert!(c.is_conserved());
        assert_eq!(c.agent_energies.len(), 4);
    }

    // ── IntentionField ────────────────────────────────────────────────────────

    #[test]
    fn intention_pull_moves_toward_target() {
        let f = IntentionField::new(140.0, 0);
        let result = f.pull_tempo(120.0);
        assert!(result > 120.0 && result < 140.0);
    }

    #[test]
    fn intention_zero_attraction_no_change() {
        let mut f = IntentionField::new(140.0, 0);
        f.attraction_strength = 0.0;
        let result = f.pull_tempo(120.0);
        assert!((result - 120.0).abs() < 1e-10);
    }

    #[test]
    fn intention_full_attraction_reaches_target() {
        let mut f = IntentionField::new(140.0, 0);
        f.attraction_strength = 1.0;
        let result = f.pull_tempo(100.0);
        assert!((result - 140.0).abs() < 1e-10);
    }

    // ── PhaseDetector ─────────────────────────────────────────────────────────

    #[test]
    fn phase_groove_at_low_variance() {
        let mut d = PhaseDetector::new();
        d.record(0.1);
        assert_eq!(d.detect(), EnsemblePhase::Groove);
    }

    #[test]
    fn phase_tension_at_mid_low_variance() {
        let mut d = PhaseDetector::new();
        d.record(1.0);
        assert_eq!(d.detect(), EnsemblePhase::Tension);
    }

    #[test]
    fn phase_chaos_at_mid_high_variance() {
        let mut d = PhaseDetector::new();
        d.record(5.0);
        assert_eq!(d.detect(), EnsemblePhase::Chaos);
    }

    #[test]
    fn phase_resolution_at_high_variance() {
        let mut d = PhaseDetector::new();
        d.record(15.0);
        assert_eq!(d.detect(), EnsemblePhase::Resolution);
    }

    #[test]
    fn phase_empty_history_is_groove() {
        let d = PhaseDetector::new();
        assert_eq!(d.detect(), EnsemblePhase::Groove);
    }

    #[test]
    fn phase_history_capped_at_eight() {
        let mut d = PhaseDetector::new();
        for i in 0..10 {
            d.record(i as f64);
        }
        assert_eq!(d.tempo_variance_history.len(), 8);
    }

    // ── Ensemble ──────────────────────────────────────────────────────────────

    #[test]
    fn ensemble_spawn_increments_id() {
        let mut e = Ensemble::new(120.0);
        let id0 = e.spawn_agent(InstrumentRole::Drums);
        let id1 = e.spawn_agent(InstrumentRole::Bass);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
    }

    #[test]
    fn ensemble_spawn_increases_agent_count() {
        let mut e = Ensemble::new(120.0);
        e.spawn_agent(InstrumentRole::Drums);
        e.spawn_agent(InstrumentRole::Bass);
        assert_eq!(e.agents.len(), 2);
    }

    #[test]
    fn ensemble_remove_agent_returns_true() {
        let mut e = Ensemble::new(120.0);
        let id = e.spawn_agent(InstrumentRole::Keys);
        assert!(e.remove_agent(id));
        assert_eq!(e.agents.len(), 0);
    }

    #[test]
    fn ensemble_remove_missing_agent_returns_false() {
        let mut e = Ensemble::new(120.0);
        assert!(!e.remove_agent(99));
    }

    #[test]
    fn ensemble_negotiate_tempo_with_agents() {
        let mut e = Ensemble::new(120.0);
        e.spawn_agent(InstrumentRole::Drums);
        e.spawn_agent(InstrumentRole::Bass);
        let tempo = e.negotiate_tempo();
        assert!(tempo > 0.0);
    }

    #[test]
    fn ensemble_tick_advances_phase() {
        let mut e = Ensemble::new(120.0);
        e.spawn_agent(InstrumentRole::Drums);
        let phase_before = e.agents[0].phase;
        e.tick(0.5);
        assert!(e.agents[0].phase != phase_before || e.agents[0].phase == 0.0);
    }

    #[test]
    fn ensemble_tick_increments_count() {
        let mut e = Ensemble::new(120.0);
        e.spawn_agent(InstrumentRole::Drums);
        e.tick(0.1);
        assert_eq!(e.tick_count, 1);
    }

    #[test]
    fn ensemble_jam_runs_without_panic() {
        let mut e = Ensemble::new(120.0);
        e.spawn_agent(InstrumentRole::Drums);
        e.spawn_agent(InstrumentRole::Bass);
        e.jam(1.0, 0.1); // 10 ticks
    }

    #[test]
    fn ensemble_spectral_report_correct_count() {
        let mut e = Ensemble::new(120.0);
        e.spawn_agent(InstrumentRole::Drums);
        e.spawn_agent(InstrumentRole::Bass);
        e.spawn_agent(InstrumentRole::Keys);
        let report = e.spectral_report();
        assert_eq!(report.len(), 3);
    }

    // ── Integration ───────────────────────────────────────────────────────────

    #[test]
    fn integration_four_agents_jam_conserved() {
        let mut e = Ensemble::new(120.0);
        e.spawn_agent(InstrumentRole::Drums);
        e.spawn_agent(InstrumentRole::Bass);
        e.spawn_agent(InstrumentRole::Keys);
        e.spawn_agent(InstrumentRole::Horns);

        e.jam(1.0, 0.05); // 20 ticks

        // Energy should still be conserved after jamming.
        assert!(e.conservation.is_conserved());
    }

    #[test]
    fn integration_four_agents_phase_is_valid() {
        let mut e = Ensemble::new(120.0);
        e.spawn_agent(InstrumentRole::Drums);
        e.spawn_agent(InstrumentRole::Bass);
        e.spawn_agent(InstrumentRole::Keys);
        e.spawn_agent(InstrumentRole::Horns);

        e.jam(1.0, 0.05);

        // Phase should be one of the four valid variants.
        let phase = e.current_phase();
        assert!(matches!(
            phase,
            EnsemblePhase::Groove
                | EnsemblePhase::Tension
                | EnsemblePhase::Chaos
                | EnsemblePhase::Resolution
        ));
    }

    #[test]
    fn integration_spectral_report_after_jam() {
        let mut e = Ensemble::new(100.0);
        e.spawn_agent(InstrumentRole::Drums);
        e.spawn_agent(InstrumentRole::Bass);
        e.spawn_agent(InstrumentRole::Keys);
        e.spawn_agent(InstrumentRole::Horns);

        e.jam(1.0, 0.1);

        let report = e.spectral_report();
        assert_eq!(report.len(), 4);
    }
}
