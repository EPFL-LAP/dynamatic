for (benchmark : benchmarks) {
  for (N : [ 0, 1, 2, 3, 4 ]) {
    for (cp = 10; cp >= 4; cp -= 0.5) {

      compile(buffer_size = 4);
      MILP_Buffer(cp);
      Heur_Remove_Buffer();

      clock_cycle = simulate();
      report(clock_cycle);

      constraint_cp = cp - 0.3;
      while (1) {
        slack = place_and_route(constraint_cp);
        if (slack > 0)
          break;
        step = max(floor(-slack * 10) / 10, 0.1);
        constraint_cp = constraint_cp + step;
      }

      report(cp, constraint_cp);
    }

    best_time = min(clock_cycle * constraint_cp);
  }
}

for (benchmark : benchmarks) {
  for (N : [ 0, 1, 2, 3, 4 ]) {

    prev_clock_cycle = inf;
    for (cp = 10;; cp -= 0.5) {
      compile(buffer_size = 4);
      MILP_Buffer(cp);

      clock_cycle = simulate();

      if (clock_cycle > 1.05 * prev_clock_cycle)
        break;
    }

    report(prev_clock_cycle);

    compile(buffer_size = 4);
    MILP_Buffer(cp + 0.5);

    Heur_Remove_Buffer();

    slack = place_and_route(constraint_cp);
    report(cp - slack);
  }
}

for (benchmark : benchmarks) {
  for (N : [ 0, 1, 2, 3, 4 ]) {

    for (fork_fifo_size : [ 10, … , 0 ]) {
      compile(buffer_size = fork_fifo_size);
      MILP_Buffer(cp = 10);

      clock_cycle = simulate();
      if (best_clock_cycle == nullptr || clock_cyle <= best_clock_cycle)
        best_clock_cycle = clock_cycle;
      best_size = fork_fifo_size;
      else break;
    }

    compile(buffer_size = best_size);

    real_buf_size = extract_log();
    remove_zero(real_buf_size);
    binary_search();
    real_buf_size = extract_log();
    resize_fifo(real_buf_size);

    for (cp = 10; cp >= 3; cp -= 0.5) {

      MILP_Buffer(cp);
      clock_cycle = simulate();

      clock_cycles.append(clock_cycle);

      constraint_cp = cp;
      while (1) {
        slack = place_and_route(constraint_cp);
        if (slack > 0)
          break;
        step = max(floor(-slack * 100) / 100, 0.1);
        constraint_cp = constraint_cp + step;
      }

      cps.append(constraint_cp);

      if (clock_cycle * constraint_cp > 1.05 * min(map(*, clock_cycles, cps)))
        break;
    }

    best_time = min(map(*, clock_cycles, cps));
  }
}
