import copy
import random
import pandas as pd
from .strategy import LSTMStratParams, LSTMStrategy
from .risk import RiskManager, RiskConfig
from .backtest import Backtester
from .utils import logger

class EvoSearch:
    def __init__(self, cfg):
        self.cfg = cfg
        wf = cfg["walk_forward"]
        self.pop = wf["pop"]
        self.gens = wf["gens"]
        self.topk = wf["topk"]
        self.rng = random.Random(1337)

    @staticmethod
    def _normalize_weights(weights: dict) -> dict:
        total = sum(weights.values())
        if total == 0:
            return {k: 1.0 / len(weights) for k in weights}
        return {k: v / total for k, v in weights.items()}

    def sample_params(self) -> LSTMStratParams:
        raw_weights = {
            "rsi": self.rng.uniform(0.1, 1.0),
            "macd": self.rng.uniform(0.1, 1.0),
            "bb": self.rng.uniform(0.1, 1.0),
        }
        normalized_weights = self._normalize_weights(raw_weights)

        return LSTMStratParams(
            prediction_threshold=round(self.rng.uniform(0.0005, 0.003), 6),  # EXTREMELY aggressive
            indicator_weights={k: round(v, 6) for k, v in normalized_weights.items()},
            exit_mode=self.rng.choice(['mechanical', 'intelligent']),
            atr_mult_sl=round(self.rng.uniform(1.5, 4.0), 6),
            initial_trail_pct=round(self.rng.uniform(0.01, 0.08), 6),
            profit_trigger_pct=round(self.rng.uniform(0.02, 0.12), 6),
            tighter_trail_pct=round(self.rng.uniform(0.005, 0.04), 6),
            lstm_disagreement_pct=round(self.rng.uniform(0.005, 0.03), 6),
            future_bars=self.rng.randint(3, 15),  # Sample future_bars

            vol_target=round(self.rng.uniform(0.5, 1.0), 6),
            max_daily_loss=round(self.rng.uniform(0.05, 0.25), 6),
            max_exposure=round(self.rng.uniform(0.5, 1.0), 6),
            risk_per_trade=round(self.rng.uniform(0.1, 0.5), 6),
        )

    def mutate(self, p: LSTMStratParams) -> LSTMStratParams:
        q = copy.deepcopy(p)
        mutable_attrs = list(q.__dict__.keys())

        attr_to_mutate = self.rng.choice(mutable_attrs)

        if attr_to_mutate == 'indicator_weights':
            key_to_mutate = self.rng.choice(list(q.indicator_weights.keys()))
            scale = 0.25
            q.indicator_weights[key_to_mutate] *= (1 + self.rng.uniform(-scale, scale))
            q.indicator_weights[key_to_mutate] = max(0.05, q.indicator_weights[key_to_mutate])
            q.indicator_weights = self._normalize_weights(q.indicator_weights)
            q.indicator_weights = {k: round(v, 6) for k, v in q.indicator_weights.items()}

        elif attr_to_mutate == 'exit_mode':
            q.exit_mode = 'intelligent' if q.exit_mode == 'mechanical' else 'mechanical'

        elif attr_to_mutate == 'future_bars':
            scale = 0.2
            new_val = int(q.future_bars * (1 + self.rng.uniform(-scale, scale)))
            q.future_bars = max(3, min(new_val, 15))

        

        elif isinstance(getattr(q, attr_to_mutate), float):
            current_val = getattr(q, attr_to_mutate)

            # Group attributes by scale for cleaner code
            scale_0025_attrs = {'prediction_threshold', 'initial_trail_pct', 'tighter_trail_pct', 'lstm_disagreement_pct'}
            scale_005_attrs = {'profit_trigger_pct', 'max_daily_loss'}
            scale_05_attrs = {'vol_target', 'max_exposure', 'risk_per_trade'}
            scale_1_attrs = set()
            scale_01_attrs = {'atr_mult_sl'}

            if attr_to_mutate in scale_0025_attrs:
                scale = 0.0025
                new_val = current_val * (1 + self.rng.uniform(-scale, scale))
                if attr_to_mutate == 'prediction_threshold':
                    new_val = max(0.0025, min(new_val, 0.03))
                elif attr_to_mutate == 'initial_trail_pct':
                    new_val = max(0.005, min(new_val, 0.1))
                elif attr_to_mutate == 'tighter_trail_pct':
                    new_val = max(0.0025, min(new_val, 0.09))
                elif attr_to_mutate == 'lstm_disagreement_pct':
                    new_val = max(0.005, min(new_val, 0.1))
            elif attr_to_mutate in scale_005_attrs:
                scale = 0.005
                new_val = current_val * (1 + self.rng.uniform(-scale, scale))
                if attr_to_mutate == 'profit_trigger_pct':
                    new_val = max(0.02, min(new_val, 0.1))
                elif attr_to_mutate == 'max_daily_loss':
                    new_val = max(0.05, min(new_val, 0.25))
            elif attr_to_mutate in scale_05_attrs:
                scale = 0.05
                new_val = current_val * (1 + self.rng.uniform(-scale, scale))
                if attr_to_mutate == 'vol_target':
                    new_val = max(0.5, min(new_val, 1.0))
                elif attr_to_mutate == 'max_exposure':
                    new_val = max(0.5, min(new_val, 0.95))
                elif attr_to_mutate == 'risk_per_trade':
                    new_val = max(0.1, min(new_val, 0.5))
            elif attr_to_mutate in scale_1_attrs:
                scale = 1.0
                new_val = current_val * (1 + self.rng.uniform(-scale, scale))
                new_val = max(2.0, min(new_val, 10.0))  # leverage in multiples of 1.0
            elif attr_to_mutate in scale_01_attrs:
                scale = 0.1
                new_val = current_val * (1 + self.rng.uniform(-scale, scale))
                new_val = max(1.0, min(new_val, 5.0))
            else:
                # Fallback for any ungrouped float attributes
                scale = 0.1
                new_val = current_val * (1 + self.rng.uniform(-scale, scale))

            q.__dict__[attr_to_mutate] = round(new_val, 6) if isinstance(new_val, float) else new_val
        
        return q

    def score(self, metrics: dict) -> float:
        if not metrics:
            return -1e9

        start_equity = self.cfg["risk"]["backtest_equity"]
        final_equity = metrics.get("Final", start_equity)

        # MASSIVE penalty for no trades
        if abs(final_equity - start_equity) < 0.001:
            return -1e6

        return final_equity

    # NEW: cache predictions once and reuse in search()
    def _attach_cached_predictions(self, features: pd.DataFrame, model) -> pd.DataFrame:
        df = features.copy()
        try:
            preds = model.predict_sequence(df)
        except Exception as e:
            logger.warning(f"Model prediction failed in optimizer: {e}")
            return df
        import numpy as np
        if len(preds) < len(df):
            padding = np.full(len(df) - len(preds), np.nan)
            preds = np.concatenate((padding, preds))
        with np.errstate(invalid='ignore', divide='ignore'):
            df['prediction'] = (preds - df['close']) / df['close']
        return df

    def search(self, search_features: pd.DataFrame, cfg, model) -> LSTMStratParams:
        logger.info("Starting EXTREMELY aggressive evolutionary search...")

        lot_size = 0.1
        # Note: risk_manager will be created per params in the loop

        # NEW: compute cached predictions once
        search_df = self._attach_cached_predictions(search_features, model)
        
        population = []
        best_overall_score = -float('inf')
        best_overall_params = None

        # Test a few manual ultra-aggressive parameters first
        manual_test_params = [
            LSTMStratParams(
                prediction_threshold=0.0001,  # 0.01%
                indicator_weights={'rsi': 0.33, 'macd': 0.33, 'bb': 0.34},
                exit_mode='mechanical'
            ),
            LSTMStratParams(
                prediction_threshold=0.0005,  # 0.05%
                indicator_weights={'rsi': 0.33, 'macd': 0.33, 'bb': 0.34},
                exit_mode='mechanical'
            ),
            LSTMStratParams(
                prediction_threshold=0.001,   # 0.1%
                indicator_weights={'rsi': 0.33, 'macd': 0.33, 'bb': 0.34},
                exit_mode='mechanical'
            )
        ]

        logger.info("Testing manual ultra-aggressive parameters...")
        for i, p in enumerate(manual_test_params):
            try:
                risk_cfg = {
                    "backtest_equity": cfg["risk"]["backtest_equity"],
                    "vol_target": p.vol_target,
                    "leverage": cfg["risk"]["leverage"],
                    "max_daily_loss": p.max_daily_loss,
                    "max_exposure": p.max_exposure,
                    "risk_per_trade": p.risk_per_trade,
                    "max_open_orders": cfg["risk"]["max_open_orders"],
                }
                risk_manager = RiskManager(RiskConfig.from_cfg(risk_cfg), lot_size=lot_size, fees=cfg["fees"])
                strategy = LSTMStrategy(p, risk_manager, model, fees=cfg["fees"])
                bt = Backtester(cfg, strategy)
                # CHANGED: use cached predictions
                metrics = bt.run(search_df)
                score = self.score(metrics)
                population.append((p, score))

                logger.info(f"Manual test {i}: pred_th={p.prediction_threshold:.6f}, "
                            f"score={score:.2f}, trades_occurred={abs(metrics.get('Final', 0) - cfg['risk']['backtest_equity']) > 0.001}")

                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_params = copy.deepcopy(p)

            except Exception as e:
                logger.warning(f"Manual test {i} failed: {e}")
                population.append((p, -1e9))

        # Add random population
        initial_params = [self.sample_params() for _ in range(self.pop - len(manual_test_params))]
        
        for i, p in enumerate(initial_params):
            try:
                risk_cfg = {
                    "backtest_equity": cfg["risk"]["backtest_equity"],
                    "vol_target": p.vol_target,
                    "leverage": cfg["risk"]["leverage"],
                    "max_daily_loss": p.max_daily_loss,
                    "max_exposure": p.max_exposure,
                    "risk_per_trade": p.risk_per_trade,
                    "max_open_orders": cfg["risk"]["max_open_orders"],
                }
                risk_manager = RiskManager(RiskConfig.from_cfg(risk_cfg), lot_size=lot_size, fees=cfg["fees"])
                strategy = LSTMStrategy(p, risk_manager, model, fees=cfg["fees"])
                bt = Backtester(cfg, strategy)
                # CHANGED: use cached predictions
                metrics = bt.run(search_df)
                score = self.score(metrics)
                population.append((p, score))

                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_params = copy.deepcopy(p)

            except Exception as e:
                logger.warning(f"Initial param {i} failed: {e}")
                population.append((p, -1e9))

        # Evolution loop
        for gen in range(self.gens):
            population.sort(key=lambda x: x[1], reverse=True)
            
            if population[0][1] > best_overall_score:
                best_overall_score = population[0][1]
                best_overall_params = copy.deepcopy(population[0][0])
            
            # Count strategies that actually traded
            trading_strategies = sum(1 for _, score in population if score > -1e5)
            
            logger.info(f"Gen {gen+1}/{self.gens}: best_score={population[0][1]:.2f}, "
                    f"trading_strategies={trading_strategies}/{len(population)}")
            
            if trading_strategies == 0:
                logger.warning("NO STRATEGIES ARE TRADING! Creating ultra-aggressive population...")
                # Create extremely aggressive new population
                new_pop = []
                for _ in range(self.pop):
                    p = LSTMStratParams(
                        prediction_threshold=self.rng.uniform(0.00001, 0.002),  # 0.001% to 0.2%
                        indicator_weights={'rsi': 0.33, 'macd': 0.33, 'bb': 0.34},
                        exit_mode='mechanical'
                    )
                    new_pop.append((p, -1e9))
                population = new_pop
                continue

            # Select elites
            elites = [p for p, s in population[:self.topk] if s > -1e5]
            if not elites:
                elites = [population[0][0]]

            new_population = population[:self.topk]
            
            # Generate children
            num_children = self.pop - len(new_population)
            
            for i in range(num_children):
                parent = self.rng.choice(elites)
                mutated_p = self.mutate(parent)
                try:
                    risk_cfg = {
                        "backtest_equity": cfg["risk"]["backtest_equity"],
                        "vol_target": mutated_p.vol_target,
                        "leverage": cfg["risk"]["leverage"],
                        "max_daily_loss": mutated_p.max_daily_loss,
                        "max_exposure": mutated_p.max_exposure,
                        "risk_per_trade": mutated_p.risk_per_trade,
                        "max_open_orders": cfg["risk"]["max_open_orders"],
                    }
                    risk_manager = RiskManager(RiskConfig.from_cfg(risk_cfg), lot_size=lot_size, fees=cfg["fees"])
                    strategy = LSTMStrategy(mutated_p, risk_manager, model, fees=cfg["fees"])
                    bt = Backtester(cfg, strategy)
                    # CHANGED: use cached predictions
                    metrics = bt.run(search_df)
                    score = self.score(metrics)
                    new_population.append((mutated_p, score))

                    if score > best_overall_score:
                        best_overall_score = score
                        best_overall_params = copy.deepcopy(mutated_p)

                except Exception as e:
                    logger.debug(f"Child {i} failed: {e}")
                    new_population.append((mutated_p, -1e9))

            population = new_population

        # Return best parameters
        final_best = best_overall_params if best_overall_params else population[0][0]
        
        if best_overall_score <= -1e5:
            logger.error("NO TRADING STRATEGIES FOUND! Using most aggressive possible parameters.")
            final_best = LSTMStratParams(
                prediction_threshold=0.00001,  # 0.001% - extremely aggressive
                indicator_weights={'rsi': 0.33, 'macd': 0.33, 'bb': 0.34},
                exit_mode='mechanical'
            )

        logger.info(f"Evolution complete. Best score: {best_overall_score:.2f}")
        logger.info(f"Final params: pred_th={final_best.prediction_threshold:.6f}")

        return final_best

    def adapt_parameters_for_equity(self, base_params: LSTMStratParams, current_equity: float) -> LSTMStratParams:
        """Adjust parameters based on current equity level for small capital strategy"""
        equity_ratio = current_equity / 25.0  # Starting capital

        adapted = copy.deepcopy(base_params)

        if equity_ratio < 2.0:  # Still small capital
            adapted.prediction_threshold *= 0.8  # More aggressive entries
            adapted.atr_mult_sl *= 1.2  # Tighter stops
        elif equity_ratio > 5.0:  # Growing capital
            adapted.prediction_threshold *= 1.2  # More conservative
            adapted.atr_mult_sl *= 0.9  # Wider stops

        return adapted

    def optimize_for_fees(self, params: LSTMStratParams, fee_rate: float = 0.0005) -> LSTMStratParams:
        """Adjust parameters to maximize profit after fees"""
        # Increase prediction threshold to account for fees
        fee_adjustment = 2 * fee_rate  # Round trip
        params.prediction_threshold += fee_adjustment

        # Reduce position sizes to minimize fee impact
        params.risk_per_trade *= 0.8

        return params
