function computeProbs() {
    let nPeasants = parseInt(document.querySelector("#inputs input[name=npeasants]").value);
    let nSoldiers = parseInt(document.querySelector("#inputs input[name=nsoldiers]").value);
    let nKnights = parseInt(document.querySelector("#inputs input[name=nknights]").value);
    let mercenary = document.querySelector("#inputs input[name=mercenary]").checked;
    // Compute single combat probabilities
    {
        let data = combatTable(nPeasants, nSoldiers, nKnights, mercenary);
        data.shift(); // Remove attack 0
        let [maxProb, cumProb, attProb, defProb] = aggregateStats(data);
        // Format output
        let table = document.querySelector("#attdef");
        table.innerHTML = "<tr><th></th><th>Att</th>" +
                data.map((_, index) => `<th>${index + 1}</th>`).join("") +
                "</tr><tr><th>Def</th><th></th>" +
                arange(data.length).map(att => {
                    let value = attProb[att];
                    let relative = 1 - value / Math.max(...attProb);
                    let color = `hsl(0deg 75% 30% / ${(1 - 0.99 * relative) * 100}%`;
                    return `<td class="bottom-border" style="background-color: ${color}">
                        <div class="tooltip" style="background-color: hsl(340deg 50% 20%);">
                            P(&#183;) = ${(value * 100).toFixed(2)}% <br />
                            P(&GreaterEqual;) = ${(cumProb[att][0] * 100).toFixed(2)}%
                        </div>
                    </td>`;
                }).join("") +
                "</tr>";
        for (let def = 0; def < data[0].length; def++) {
            let value = defProb[def];
            let relative = 1 - value / Math.max(...defProb);
            let color = `hsl(240deg 75% 30% / ${(1 - 0.99 * relative) * 100}%`;
            let defCell = `<td class="right-border" style="background-color: ${color}">
                <div class="tooltip" style="background-color: hsl(260deg 50% 20%);">
                    P(&#183;) = ${(value * 100).toFixed(2)}% <br />
                    P(&GreaterEqual;) = ${(cumProb[0][def] * 100).toFixed(2)}%
                </div>
            </td>`;
            table.innerHTML += `<tr><th>${def}</th>${defCell}` +
                    arange(data.length).map(att => {
                        let value = data[att][def];
                        let relative = 1 - value / maxProb;
                        let color = `hsl(160deg 30% 25% / ${(1 - 0.99 * relative) * 100}%`;
                        return `<td style="background-color: ${color}">
                            <div class="tooltip" style="background-color: hsl(300deg 10% 25%);">
                                P(&#183;) = ${(value * 100).toFixed(2)}% <br />
                                P(&GreaterEqual;) = ${(cumProb[att][def] * 100).toFixed(2)}%
                            </div>
                        </td>`;
                    }).join("") +
                    "</tr>";
        }
    }
    // Compute final outcome
    let enemyDamage = parseInt(document.querySelector("#inputs input[name=eattack]").value);
    let enemyHealth = parseInt(document.querySelector("#inputs input[name=edefense]").value);
    {
        let data = combatOutcomes(enemyDamage, enemyHealth, nPeasants, nSoldiers, nKnights, mercenary);
        let maxProb = Math.max(...data);
        let cumProb = data.reduceRight((a, x) => [x + (a[0] || 0), ...a], []);
        // Format output
        let table = document.querySelector("#outcome");
        table.innerHTML = "<tr><th colspan='2' title='Units remaining after combat' style='cursor: help;'>Units</th></tr>" +
                data.map((value, index) => {
                    let baseColor = "140deg 50%";
                    if (index == 0) {
                        baseColor = "0deg 90%";
                    } else if (index - 1 < nKnights) {
                        baseColor = "300deg 10%";
                    } else if (index - 1 < nKnights + nSoldiers) {
                        baseColor = "240deg 40%";
                    }
                    let relative = 1 - value / maxProb;
                    let color = `hsl(${baseColor} 30% / ${(1 - 0.99 * relative) * 100}%`;
                    return `<tr><th>${index ? index - 1 : "&#9760;"}</th><td style="background-color: ${color}">
                        <div class="tooltip" style="background-color: hsl(0deg 0% 20%);">
                            P(&#183;) = ${(value * 100).toFixed(2)}% <br />
                            P(&GreaterEqual;) = ${(cumProb[index] * 100).toFixed(2)}%
                        </div>
                    </td></tr>`;
                }).join("");
    }
}


function adjustCount(element) {
    // TODO: Ensure total counts are valid
}

// Calculations

function aggregateStats(data) {
    let maxProb = data.reduce((a, x) => Math.max(a, x.reduce((a, x) => Math.max(a, x), 0)), 0);
    let cumProb = data.reduceRight((a, x) => [
        x.reduceRight((a, x) => [x + (a[0] || 0), ...a], []).map((x, i) => x + ((a[0] || [])[i] || 0)),
        ...a
    ], []);
    let attProb = data.map(x => x.reduce((a, x) => a + x));
    let defProb = data.reduce((a, x) => x.map((x, i) => x + (a[i] || 0)), []);
    return [maxProb, cumProb, attProb, defProb];
}

function combatOutcomes(enemyDamage, enemyHealth, nPeasants, nSoldiers, nKnights, mercenary = false) {
    // DP with enemyHealth and playerHealth, each cell is distribution over final outcome
    const playerHealth = nPeasants + nSoldiers + nKnights;
    const units = pH => {
        let nK = Math.min(nKnights, pH);
        let nS = Math.min(nSoldiers, pH - nK);
        let nP = Math.min(nPeasants, pH - nK - nS);
        return [nP, nS, nK];
    };
    let instances = new Array(enemyHealth + 1).fill(0).map(() => new Array(playerHealth + 1).fill(0));
    // Base cases
    for (let eH = 1; eH <= enemyHealth; eH++) {
        instances[eH][0] = [1, ...new Array(playerHealth + 1).fill(0)]; // Failure case
    }
    for (let pH = 0; pH <= playerHealth; pH++) {
        instances[0][pH] = [0, ...arange(playerHealth + 1).map(i => 0 + (i == pH))]; // Winning case
    }
    // Recursive step
    for (let pH = 1; pH <= playerHealth; pH++) {
        for (let eH = 1; eH <= enemyHealth; eH++) {
            let dist = arange(playerHealth + 2).fill(0);
            // Iterate over all possibilities
            let data = combatTable(...units(pH), mercenary);
            for (let att = 1; att < data.length; att++) {
                for (let def = 0; def < data[0].length; def++) {
                    let remEnemyHealth = Math.max(eH - att, 0);
                    let remPlayerHealth = Math.max(pH - Math.max(enemyDamage - def, 0), 0);
                    // Sum
                    dist = dist.map((x, i) => x + instances[remEnemyHealth][remPlayerHealth][i] * data[att][def]);
                }
            }
            instances[eH][pH] = dist;
        }
    }
    return instances[enemyHealth][playerHealth];
}


// Probabilities

function arange(n) {
    return new Array(n).fill(0).map((_, index) => index);
}

function nCr(n, r) {
    let result = 1;
    for (let i = n; i > n - r; i--) {
        result *= i;
    }
    for (let i = 1; i <= r; i++) {
        result /= i;
    }
    return result;
}

function binom(n, r, p) {
    return nCr(n, r) * Math.pow(p, r) * Math.pow(1 - p, n - r);
}

function combatTable(nPeasants, nSoldiers, nKnights = 0, mercenary = false) {
    let maxDamage = nPeasants + nSoldiers + 2 * nKnights + 3;
    let maxDefend = nPeasants + nSoldiers + 2 * nKnights;
    let instances = new Array(maxDamage + 1).fill(0).map(() => new Array(maxDefend + 1).fill(0));
    // Compute probabilities
    // Peasants
    for (let pAtt = 0; pAtt <= nPeasants; pAtt++) {
        let probPAtt = binom(nPeasants, pAtt, 1/6);
        for (let pDef = 0; pDef <= nPeasants - pAtt; pDef++) {
            let probPDef = binom(nPeasants - pAtt, pDef, 1/5);
            // Soldiers
            for (let sAtt = 0; sAtt <= nSoldiers; sAtt++) {
                let probSAtt = binom(nSoldiers, sAtt, 1/3);
                for (let sDef = 0; sDef <= nSoldiers - sAtt; sDef++) {
                    let probSDef = binom(nSoldiers - sAtt, sDef, 1/2);
                    // Knights
                    for (let kTotal = 0; kTotal <= nKnights; kTotal++) {
                        let probKTotal = binom(nKnights, kTotal, 2/3);
                        for (let kAtt = 0; kAtt <= 2 * kTotal; kAtt++) {
                            let probKAtt = binom(2 * kTotal, kAtt, 1/2);
                            kDef = 2 * kTotal - kAtt;
                            // Leader
                            for (let playerAtt of [1, 2, 3]) {
                                let probPlayerAtt = (mercenary ? [0, 1/2, 1/3, 1/6] : [0, 5/6, 0, 1/6])[playerAtt];
                                // Compute total attack and defense
                                let totalAtt = pAtt + sAtt + kAtt + playerAtt;
                                let totalDef = pDef + sDef + kDef;
                                let totalProb = probPAtt * probPDef * probSAtt * probSDef *
                                        probKTotal * probKAtt * probPlayerAtt;
                                instances[totalAtt][totalDef] += totalProb;
                            }
                        }
                    }
                }
            }
        }
    }
    return instances;
}